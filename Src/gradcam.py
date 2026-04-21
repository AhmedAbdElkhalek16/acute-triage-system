"""
gradcam.py
----------
Grad-CAM implementation for visualizing model decisions.
Produces a heatmap overlay showing WHERE in the image
the model focused to make its prediction.

This is critical for medical AI — doctors need to see
*why* the model flagged something, not just that it did.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────
#  Grad-CAM Core
# ─────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into the last convolutional layer of a model,
    captures gradients during backward pass, and builds
    a spatial importance map.

    Usage:
        cam    = GradCAM(model, target_layer='backbone.conv_head')
        image  = preprocess(img)                    # (1, 3, H, W)
        heatmap, overlay = cam.generate(image, class_idx=1)
        cv2.imwrite('heatmap.png', overlay)
    """

    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Args:
            model        : trained PyTorch model
            target_layer : dot-notation path to the conv layer
                           EfficientNet : 'backbone.conv_head'
                           DenseNet-121 : 'backbone.features.denseblock4'
        """
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []

        self._register_hooks()

    def _get_layer(self) -> torch.nn.Module:
        """Traverse model by dot-separated attribute path."""
        layer = self.model
        for attr in self.target_layer.split('.'):
            layer = getattr(layer, attr)
        return layer

    def _register_hooks(self):
        layer = self._get_layer()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self,
                 image     : torch.Tensor,
                 class_idx : int = None,
                 original_image: np.ndarray = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap.

        Args:
            image          : preprocessed tensor (1, 3, H, W)
            class_idx      : target class (None → argmax / predicted class)
            original_image : BGR numpy array for overlay (optional)

        Returns:
            heatmap  : (H, W) float32 array, 0–1
            overlay  : (H, W, 3) uint8 BGR array with heatmap drawn on image
        """
        self.model.eval()
        image = image.requires_grad_(True)

        # Forward pass
        logits = self.model(image)
        probs  = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Pool gradients across spatial dimensions → (C,)
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])

        # Weight activations by pooled gradients → (C, H, W)
        activations = self.activations.squeeze(0)
        for i, w in enumerate(pooled_grads):
            activations[i] *= w

        # Average over channels → (H, W)
        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)                     # ReLU
        heatmap = heatmap / (heatmap.max() + 1e-8)           # Normalize 0–1

        # Build overlay
        if original_image is not None:
            overlay = self._overlay_heatmap(heatmap, original_image)
        else:
            overlay = self._heatmap_to_bgr(heatmap)

        confidence = float(probs[0, class_idx].item())
        return heatmap, overlay, confidence

    @staticmethod
    def _heatmap_to_bgr(heatmap: np.ndarray,
                        size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Convert normalized float heatmap → colorized BGR image."""
        h = cv2.resize(heatmap, size)
        h = (h * 255).astype(np.uint8)
        return cv2.applyColorMap(h, cv2.COLORMAP_JET)

    @staticmethod
    def _overlay_heatmap(heatmap : np.ndarray,
                         image   : np.ndarray,
                         alpha   : float = 0.4) -> np.ndarray:
        """
        Blend heatmap onto original image.
        alpha = heatmap opacity (0.4 = 40% heatmap, 60% original).
        """
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8   = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3 and image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
        return overlay


# ─────────────────────────────────────────────
#  Target Layer Map
# ─────────────────────────────────────────────

TARGET_LAYERS = {
    'efficientnet_b4' : 'backbone.conv_head',
    'densenet121'     : 'backbone.features.norm5',
}


def get_gradcam(model, model_name: str) -> GradCAM:
    """
    Convenience factory — picks the right target layer for each backbone.

    Args:
        model      : trained model instance
        model_name : 'efficientnet_b4' or 'densenet121'
    """
    if model_name not in TARGET_LAYERS:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from: {list(TARGET_LAYERS.keys())}")
    return GradCAM(model, target_layer=TARGET_LAYERS[model_name])


# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import timm
    import torch.nn as nn

    print("Building a tiny EfficientNet for Grad-CAM test...")

    # Minimal model for testing (no real weights)
    backbone = timm.create_model('efficientnet_b4', pretrained=False,
                                 num_classes=0, global_pool='avg')

    class _TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(backbone.num_features, 3)
        def forward(self, x):
            return self.head(self.backbone(x))

    model = _TestModel()
    cam   = GradCAM(model, target_layer='backbone.conv_head')

    dummy = torch.randn(1, 3, 512, 512)
    heatmap, overlay, conf = cam.generate(dummy, class_idx=1)

    print(f"  Heatmap shape  : {heatmap.shape}")
    print(f"  Overlay shape  : {overlay.shape}")
    print(f"  Confidence     : {conf:.3f}")
    print("Grad-CAM OK!")

    cam.remove_hooks()