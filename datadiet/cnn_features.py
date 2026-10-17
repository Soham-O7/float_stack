"""OPTIONAL (needs torch + torchvision). Turns images into feature vectors so the same engine works on MNIST / CIFAR-10.

A CNN = conv layers (find patterns) + pooling (shrink maps, keep the strongest/average signal).
Global average pooling at the end collapses each feature map to ONE number -> one fixed-length vector per image.
That vector is what embed() / score_dataset() consume. We use it only as a frozen feature extractor: no training here.
"""
import numpy as np


def cnn_features(images, pool="avg", batch=256, device=None):
    """images: float array (N,H,W) grayscale or (N,H,W,3) RGB scaled 0..1. pool: 'avg' or 'max'."""
    import torch, torchvision
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)  # downloads weights once
    body = torch.nn.Sequential(*list(net.children())[:-2]).eval().to(device)               # drop avgpool + fc
    pool_layer = torch.nn.AdaptiveAvgPool2d(1) if pool == "avg" else torch.nn.AdaptiveMaxPool2d(1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.as_tensor(images[i:i + batch], dtype=torch.float32, device=device)
            x = x.unsqueeze(-1).repeat(1, 1, 1, 3) if x.ndim == 3 else x
            x = x.permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(x, size=64, mode="bilinear")  # small = fast; raise for accuracy
            out.append(pool_layer(body((x - mean) / std)).flatten(1).cpu().numpy())
    return np.vstack(out)
