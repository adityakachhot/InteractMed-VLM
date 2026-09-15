"""
vision_encoder/encoder.py
=========================
Visual feature extraction module for InteractMed-VLM.

Input/Output Contract:
----------------------
Input:  CXR image (torch.Tensor [B, C, H, W], PIL.Image.Image, or file path str)
Output: Visual feature tensor of shape [B, feature_dim, H_feat, W_feat]

UPDATE (Review 3): this now supports a real chest-X-ray-domain-pretrained
backbone via the TorchXRayVision library (pip install torchxrayvision),
selected through configs/model_config.yaml -> vision_encoder.model_name_or_path.

  - "torchxrayvision/densenet121-res224-all" (or any other torchxrayvision
    weights string, e.g. "densenet121-res224-chex") loads a real DenseNet121
    trained on NIH / CheXpert / MIMIC-CH / Google OpenI / RSNA chest X-rays.
  - Anything else (including the original "placeholder_vision_encoder" /
    "TODO_USER_CONFIRM_VISION_ENCODER") falls back to the original CPU stub,
    so nothing breaks if torchxrayvision isn't installed in a given environment.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image

try:
    import torchxrayvision as xrv
    import numpy as np
    _TORCHXRAYVISION_AVAILABLE = True
except ImportError:
    _TORCHXRAYVISION_AVAILABLE = False


class VisionEncoder(nn.Module):
    """
    Vision Encoder backbone for extracting dense spatial representations from CXR images.

    Candidate backbones considered (see proposal Section 4.1):
      1. 'microsoft/BiomedVLP-BioViL-T' (domain-specific chest pathology ViT)
      2. 'stanford-crfm/CheXzero' (zero-shot CXR image encoder)
      3. 'torchxrayvision/densenet121-res224-all' (DenseNet-121 trained on
         NIH/CheXpert/MIMIC-CH/OpenI/RSNA chest X-rays) <- currently wired in
      4. 'openai/clip-vit-base-patch32' (general VLM baseline)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.image_size: Tuple[int, int] = tuple(self.config.get("image_size", (224, 224)))
        self.model_name: str = self.config.get("model_name_or_path", "placeholder_vision_encoder")

        self.use_torchxrayvision = (
            _TORCHXRAYVISION_AVAILABLE and self.model_name.startswith("torchxrayvision/")
        )

        if self.use_torchxrayvision:
            weights_name = self.model_name.split("/", 1)[1]  # e.g. "densenet121-res224-all"
            self.backbone = xrv.models.DenseNet(weights=weights_name)
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False  # frozen backbone, per proposal Section 8.1/8.2
            # DenseNet121's final feature map before pooling is [B, 1024, 7, 7]
            self.feature_dim: int = 1024
            self.output_spatial_dim: Tuple[int, int] = (7, 7)
        else:
            self.feature_dim = int(self.config.get("feature_dim", 512))
            self.output_spatial_dim = tuple(self.config.get("output_spatial_dim", (7, 7)))
            # Stub projection layer for CPU unit testing and dimension alignment
            self.stub_projection = nn.Sequential(
                nn.AdaptiveAvgPool2d(self.output_spatial_dim),
                nn.Conv2d(3, self.feature_dim, kernel_size=1, bias=False),
            )

    def preprocess(self, image: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """
        Preprocesses an input image into a standardized tensor [B, 3, H, W]
        (or, for the torchxrayvision path, the single-channel tensor that
        backbone expects — handled separately in _preprocess_xrv).
        """
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
            return self._pil_to_tensor(pil_img)
        elif isinstance(image, Image.Image):
            return self._pil_to_tensor(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.size(1) == 1:
                image = image.repeat(1, 3, 1, 1)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Helper to convert PIL Image to tensor [1, 3, H, W]."""
        resized = pil_img.resize(self.image_size)
        tensor = torch.tensor(list(resized.getdata()), dtype=torch.float32).view(
            resized.size[1], resized.size[0], len(resized.getbands())
        )
        tensor = tensor.permute(2, 0, 1) / 255.0
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.size(0) > 3:
            tensor = tensor[:3, :, :]
        return tensor.unsqueeze(0)

    def _preprocess_xrv(self, image: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """torchxrayvision expects a single-channel image normalized to [-1024, 1024]."""
        if isinstance(image, str):
            pil_img = Image.open(image).convert("L")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("L")
        else:
            # Already a tensor - fall back to averaging channels
            arr = image.squeeze(0).mean(dim=0).cpu().numpy() * 255.0
            img = xrv.datasets.normalize(arr, 255)
            img = img[None, ...]
            transform = self._xrv_transform()
            img = transform(img)
            return torch.from_numpy(img).unsqueeze(0).float()

        pil_img = pil_img.resize((224, 224))
        arr = np.array(pil_img).astype(np.float32)
        img = xrv.datasets.normalize(arr, 255)
        img = img[None, ...]  # add channel dim -> [1, H, W]
        transform = self._xrv_transform()
        img = transform(img)
        return torch.from_numpy(img).unsqueeze(0).float()  # [1, 1, 224, 224]

    def _xrv_transform(self):
        return xrv.datasets.XRayCenterCrop()

    def forward(self, image: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """
        Encodes an input CXR into visual features.

        Returns:
            Visual feature tensor of shape [B, feature_dim, H_feat, W_feat].
        """
        if self.use_torchxrayvision:
            x = self._preprocess_xrv(image)
            with torch.no_grad():
                features = self.backbone.features(x)  # [1, 1024, 7, 7]
            return features

        x = self.preprocess(image)
        features = self.stub_projection(x)
        return features

    def encode(self, image: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(image)

    def predict_pathologies(self, image: Union[torch.Tensor, Image.Image, str]) -> Optional[Dict[str, float]]:
        """
        Bonus: if using the torchxrayvision backbone, also expose its native
        18-pathology classification head — useful for sanity-checking that
        the real model is actually seeing something clinically meaningful,
        independent of the (still-stub) report generator / triage classifier.
        Returns None when not using the torchxrayvision backbone.
        """
        if not self.use_torchxrayvision:
            return None
        x = self._preprocess_xrv(image)
        with torch.no_grad():
            preds = self.backbone(x).cpu()
        return {
            k: float(v)
            for k, v in zip(xrv.datasets.default_pathologies, preds[0].detach().numpy())
        }
