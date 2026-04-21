"""
models.py
---------
Two specialized models with transfer learning:
  - XRayModel  : EfficientNet-B4  → Pneumothorax / Pneumonia
  - CTModel    : DenseNet-121     → Hemorrhage / Embolism

Both inherit from a shared BaseTriageModel that handles
saving, loading, and freezing/unfreezing layers.
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path


# ─────────────────────────────────────────────
#  Base class
# ─────────────────────────────────────────────

class BaseTriageModel(nn.Module):
    """
    Shared utilities for all triage models.
    Subclasses only need to define self.backbone and self.head.
    """

    def freeze_backbone(self):
        """Freeze all backbone weights — train only the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[{self.__class__.__name__}] Backbone frozen.")

    def unfreeze_backbone(self, last_n_blocks: int = 2):
        """
        Unfreeze the last N blocks for fine-tuning.
        Call this after initial head-only training.
        """
        # Unfreeze everything first, then selectively re-freeze
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Re-freeze early layers (backbone-specific; works for timm models)
        children = list(self.backbone.children())
        for child in children[:-last_n_blocks]:
            for param in child.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[{self.__class__.__name__}] Unfroze last {last_n_blocks} blocks. "
              f"Trainable params: {trainable:,}")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[{self.__class__.__name__}] Saved → {path}")

    def load(self, path: str, device: str = 'cpu'):
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)
        print(f"[{self.__class__.__name__}] Loaded ← {path}")
        return self

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class probabilities (softmax applied)."""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.softmax(logits, dim=1)


# ─────────────────────────────────────────────
#  X-Ray Model  (EfficientNet-B4)
# ─────────────────────────────────────────────

class XRayModel(BaseTriageModel):
    """
    EfficientNet-B4 fine-tuned for chest X-Ray triage.

    Classes:
        0 → Normal
        1 → Pneumothorax   (CRITICAL)
        2 → Pneumonia      (HIGH)

    Usage:
        model = XRayModel(num_classes=3, pretrained=True)
        model.freeze_backbone()
        # ... train head for 5 epochs ...
        model.unfreeze_backbone(last_n_blocks=2)
        # ... fine-tune for 10 more epochs ...
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        # Load EfficientNet-B4 from timm (ImageNet pretrained)
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,          # Remove original classifier
            global_pool='avg',      # Global average pooling → flat vector
        )
        in_features = self.backbone.num_features  # 1792 for B4

        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, 1792)
        return self.head(features)    # (B, num_classes)


# ─────────────────────────────────────────────
#  CT Model  (DenseNet-121)
# ─────────────────────────────────────────────

class CTModel(BaseTriageModel):
    """
    DenseNet-121 fine-tuned for CT scan triage.
    DenseNet is preferred for CT because its dense connections
    help capture subtle lesion features.

    Classes:
        0 → Normal
        1 → Intracranial Hemorrhage  (CRITICAL)
        2 → Pulmonary Embolism       (HIGH)
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 dropout: float = 0.4):
        super().__init__()

        self.backbone = timm.create_model(
            'densenet121',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
        )
        in_features = self.backbone.num_features  # 1024 for DenseNet-121

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# ─────────────────────────────────────────────
#  Model Factory
# ─────────────────────────────────────────────

def get_model(modality: str,
              num_classes: int = 3,
              pretrained: bool = True,
              weights_path: str = None,
              device: str = 'cpu') -> BaseTriageModel:
    """
    Factory function — returns the right model for a given modality.

    Args:
        modality      : 'xray' or 'ct'
        num_classes   : number of output classes
        pretrained    : use ImageNet weights
        weights_path  : path to saved .pth checkpoint (optional)
        device        : 'cuda' or 'cpu'

    Returns:
        model ready for training or inference
    """
    if modality == 'xray':
        model = XRayModel(num_classes=num_classes, pretrained=pretrained)
    elif modality == 'ct':
        model = CTModel(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown modality '{modality}'. Use 'xray' or 'ct'.")

    if weights_path:
        model.load(weights_path, device=device)

    return model.to(device)


# ─────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    dummy = torch.randn(2, 3, 512, 512).to(device)

    print("── X-Ray Model (EfficientNet-B4) ──")
    xray_model = get_model('xray', device=device)
    out = xray_model(dummy)
    print(f"  Output shape : {out.shape}")    # (2, 3)
    print(f"  Probabilities: {xray_model.predict_proba(dummy)}")

    print("\n── CT Model (DenseNet-121) ──")
    ct_model = get_model('ct', device=device)
    out = ct_model(dummy)
    print(f"  Output shape : {out.shape}")

    total = sum(p.numel() for p in xray_model.parameters())
    print(f"\nX-Ray model total params: {total:,}")
    