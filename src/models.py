import torch
import torch.nn as nn
import torchvision.models as tvm

from config import CFG

#Conv-LSTM Classifier on Diffusion Activations

class ConvLSTMClassifier(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1280):
        super().__init__()

        hidden = CFG.train.lstm_hidden

        self.conv_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )

        # Bidirectional LSTM over spatial sequence
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=CFG.train.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=CFG.train.lstm_dropout if CFG.train.lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden * 2   # bidirectional

        # Attention pooling: learn which spatial positions matter for classification
        self.attn = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(CFG.train.classifier_dropout),
            nn.Linear(lstm_out_dim, 256),
            nn.GELU(),
            nn.Dropout(CFG.train.classifier_dropout / 2),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, C) → logits: (B, num_classes)
        # Conv projection: transpose to (B, C, seq_len) for Conv1d
        x = self.conv_proj(x.transpose(1, 2))   # (B, hidden, seq_len)
        x = x.transpose(1, 2)                    # (B, seq_len, hidden)

        # LSTM
        lstm_out, _ = self.lstm(x)               # (B, seq_len, hidden*2)

        # Attention pooling
        attn_w = self.attn(lstm_out)             # (B, seq_len, 1)
        attn_w = torch.softmax(attn_w, dim=1)
        context = (lstm_out * attn_w).sum(dim=1) # (B, hidden*2)

        return self.head(context)                # (B, num_classes)


# MLP on pooled activations (linear probe) :

class DiffusionMLPClassifier(nn.Module):

    def __init__(self, num_classes: int, feature_dim: int = None):
        super().__init__()
        dim = feature_dim or CFG.diffusion.feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ResNet50 Baseline : (on raw images)

class ResNet50Baseline(nn.Module):
    # ResNet50 finetuned on raw paintings.
    # Starts with only layer4 + head unfrozen; full bacbone unfrozen after warmup.

    def __init__(self, num_classes: int):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
        backbone = tvm.resnet50(weights=weights)

        # Freeze all bacbone params initially
        for p in backbone.parameters():
            p.requires_grad = False

        # Replace classification head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone

        # Immediately unfreeze layer4 + fc
        self._unfreeze_stage("layer4")

    def _unfreeze_stage(self, *stage_names: str):
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(s) for s in stage_names) or "fc" in name:
                param.requires_grad = True

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(model_type: str, num_classes: int) -> nn.Module:
    # model_type: "convlstm" | "mlp" | "resnet50"
    if model_type == "convlstm":
        return ConvLSTMClassifier(num_classes=num_classes)
    elif model_type == "mlp":
        return DiffusionMLPClassifier(num_classes=num_classes)
    elif model_type == "resnet50":
        return ResNet50Baseline(num_classes=num_classes)
    else:
        raise ValueError(f"Unnown model_type: {model_type}")
