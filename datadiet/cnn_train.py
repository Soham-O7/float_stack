"""CNN training with selectable POOLING and OPTIMIZER (PyTorch). Works for MNIST (1x28x28), CIFAR-10 (3x32x32), digits (1x8x8)."""
import time
import numpy as np
import torch
import torch.nn as nn

POOLS = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}


class SimpleCNN(nn.Module):
    """[conv-BN-ReLU-pool] x2 -> global average pool -> linear.
    conv  : slides small filters over the image to detect edges/strokes/shapes
    pool  : shrinks each map by 2x (max = keep strongest response, avg = keep mean response)
    global avg pool : one number per feature map, so the classifier does not care about image size
    """
    def __init__(self, in_ch=1, n_classes=10, pool="max", width=32):
        super().__init__()
        P = POOLS[pool]
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), P(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(), P(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 2, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def make_optimizer(name, params, lr):
    name = name.lower()
    if name == "adam":    return torch.optim.Adam(params, lr=lr)
    if name == "adamw":   return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    if name == "sgd":     return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop": return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(name)

DEFAULT_LR = {"adam": 1e-3, "adamw": 1e-3, "sgd": 0.05, "rmsprop": 1e-3}


def to_images(X, shape):
    """flat (N, H*W*C) -> torch (N, C, H, W). shape = (C, H, W)."""
    return torch.as_tensor(np.asarray(X, dtype=np.float32)).reshape(len(X), *shape)


def train_cnn(Xtr, ytr, Xte, yte, shape=(1, 28, 28), pool="max", optimizer="adam",
              epochs=5, batch=128, lr=None, seed=0, device=None):
    """Returns dict(accuracy, seconds, model). Time covers training only (same clock as the proof run)."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = int(max(np.max(ytr), np.max(yte))) + 1
    model = SimpleCNN(shape[0], n_classes, pool).to(device)
    opt = make_optimizer(optimizer, model.parameters(), lr or DEFAULT_LR[optimizer.lower()])
    lossf = nn.CrossEntropyLoss()
    Xt, yt = to_images(Xtr, shape).to(device), torch.as_tensor(np.asarray(ytr), dtype=torch.long).to(device)

    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(yt), device=device)
        for i in range(0, len(yt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(Xt[b]), yt[b]).backward()
            opt.step()
    if device == "cuda": torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    return {"accuracy": evaluate(model, Xte, yte, shape, device), "seconds": secs, "model": model}


@torch.no_grad()
def evaluate(model, X, y, shape, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    Xe = to_images(X, shape)
    pred = torch.cat([model(Xe[i:i + 1024].to(device)).argmax(1).cpu() for i in range(0, len(Xe), 1024)])
    return float((pred.numpy() == np.asarray(y)).mean())


@torch.no_grad()
def cnn_embed(model, X, shape, batch=1024):
    """Penultimate (pooled) features of a trained CNN -> better 'looks alike' vectors for scoring than raw pixels."""
    device = next(model.parameters()).device
    model.eval()
    Xe = to_images(X, shape)
    return torch.cat([model.features(Xe[i:i + batch].to(device)).flatten(1).cpu() for i in range(0, len(Xe), batch)]).numpy()
