"""
detector.py - the shared core of the project (used by train.py and app.py).

    DeepfakeDetector   light ImageNet-pretrained backbone + two small heads
                         fake head   : real photo (0) vs AI-generated (1)
                         source head : which generator the image most resembles
    train_transform    preprocessing + "social-feed" data augmentation
    eval_transform     preprocessing only (optionally with a fixed degradation)
    load_detector      rebuild a trained model from a checkpoint
    analyze            one image -> probability, likely generator, Grad-CAM heat-map
"""
import io
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import models
from torchvision import transforms as T

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics

# The five generators in the dataset: file-name prefix -> display name.
SOURCES = {"sd21": "Stable Diffusion 2.1", "sdxl": "SDXL", "sd3": "Stable Diffusion 3",
           "dalle3": "DALL-E 3", "midjourney": "Midjourney v6"}
SOURCE_KEYS = list(SOURCES)

# Light backbones: name -> (torchvision constructor, channels of the last feature map).
BACKBONES = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, 576),
    "mobilenet_v3_large": (models.mobilenet_v3_large, 960),
    "mobilenet_v2": (models.mobilenet_v2, 1280),
    "efficientnet_b0": (models.efficientnet_b0, 1280),
}


# ----------------------------------------------------------------------------- model
class DeepfakeDetector(nn.Module):
    """Pretrained convolutional features -> global average pool -> two linear heads."""

    def __init__(self, backbone="mobilenet_v3_large", pretrained=True):
        super().__init__()
        build, channels = BACKBONES[backbone]
        self.backbone_name = backbone
        self.features = build(weights="DEFAULT" if pretrained else None).features
        self.dropout = nn.Dropout(0.2)
        self.fake_head = nn.Linear(channels, 1)
        self.source_head = nn.Linear(channels, len(SOURCES))

    def heads(self, feature_map):
        pooled = self.dropout(feature_map.mean(dim=(2, 3)))
        return self.fake_head(pooled).squeeze(1), self.source_head(pooled)

    def forward(self, x):
        """Returns (fake logit [B], generator logits [B, 5])."""
        return self.heads(self.features(x))

    def freeze_backbone(self, frozen=True):
        for p in self.features.parameters():
            p.requires_grad = not frozen


# ----------------------------------------------------------------------------- transforms
def degrade(img, scale=None, blur=None, jpeg=None):
    """What a social feed does to a picture: shrink it, soften it, re-compress it."""
    if scale:
        w, h = img.size
        small = img.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
        img = small.resize((w, h), Image.BICUBIC)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if jpeg:
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=int(jpeg))
        img = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return img


class FeedAugment:
    """Random version of `degrade`, applied during training."""

    def __call__(self, img):
        return degrade(
            img,
            scale=random.uniform(0.4, 0.9) if random.random() < 0.3 else None,
            blur=random.uniform(0.3, 1.5) if random.random() < 0.2 else None,
            jpeg=random.randint(30, 95) if random.random() < 0.5 else None,
        )


def eval_transform(**degradation):
    """Resize + centre crop. Pass scale= / blur= / jpeg= to test robustness."""
    steps = [T.Resize(256), T.CenterCrop(IMG_SIZE)]
    if degradation:
        steps.append(partial(degrade, **degradation))
    return T.Compose(steps + [T.ToTensor(), T.Normalize(MEAN, STD)])


def train_transform(augment=True):
    if not augment:
        return eval_transform()
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.3, 1.0), ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        FeedAugment(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


# ----------------------------------------------------------------------------- inference
def load_detector(path="models/detector.pt", device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = DeepfakeDetector(checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def analyze(model, image, heatmap=True):
    """
    Classify one PIL image.

    Returns {"p_fake": probability the image is AI-generated,
             "sources": {generator name: probability},
             "heatmap": PIL image showing the regions behind the verdict (Grad-CAM)}.
    """
    device = next(model.parameters()).device
    crop = T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE)])(image.convert("RGB"))
    x = T.Normalize(MEAN, STD)(T.ToTensor()(crop)).unsqueeze(0).to(device)

    with torch.set_grad_enabled(heatmap):
        feature_map = model.features(x)
        fake_logit, source_logits = model.heads(feature_map)
    p_fake = torch.sigmoid(fake_logit).item()
    source_probs = torch.softmax(source_logits, dim=1)[0].tolist()
    result = {"p_fake": p_fake, "sources": dict(zip(SOURCES.values(), source_probs)), "heatmap": None}

    if heatmap:
        # Grad-CAM: weight each feature channel by how much it supports the verdict.
        verdict_score = fake_logit if p_fake >= 0.5 else -fake_logit
        gradient = torch.autograd.grad(verdict_score.sum(), feature_map)[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=IMG_SIZE, mode="bilinear", align_corners=False)[0, 0]
        cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
        result["heatmap"] = _overlay(crop, cam)
    return result


def _overlay(image, cam, strength=0.5):
    """Blend a [0, 1] heat-map over an image (blue = ignored, red = decisive)."""
    from matplotlib import colormaps
    colour = colormaps["jet"](cam)[..., :3]
    base = np.asarray(image, dtype=np.float32) / 255.0
    alpha = strength * cam[..., None]
    blended = (1 - alpha) * base + alpha * colour
    return Image.fromarray((blended * 255).astype(np.uint8)).resize((2 * IMG_SIZE, 2 * IMG_SIZE), Image.BICUBIC)
