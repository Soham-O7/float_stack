"""Getting data in: CSV, numpy binaries, sklearn digits, MNIST/CIFAR-10 (via torchvision), and the PDF's noise injection."""
import numpy as np


def load_csv(path, label_col):
    import pandas as pd
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"label column '{label_col}' not found")   # the 'file valid?' decision in the process flow
    return df.drop(columns=[label_col]).to_numpy(dtype=float), df[label_col].to_numpy()


def load_numpy(x_path, y_path):
    """.npy files, or .npz (pass 'file.npz:key')."""
    def _get(p):
        if ":" in p:
            f, k = p.split(":"); return np.load(f)[k]
        return np.load(p)
    X, y = _get(x_path), _get(y_path)
    return X.reshape(len(X), -1).astype(float), y.ravel()


def load_digits():
    from sklearn.datasets import load_digits as _ld
    d = _ld(); return d.data, d.target


def load_torchvision(name="MNIST", root="./data", train=True):
    """MNIST / CIFAR10 as flat arrays (needs `pip install torchvision` and internet for the first download)."""
    from torchvision import datasets
    ds = getattr(datasets, {"MNIST": "MNIST", "CIFAR10": "CIFAR10"}[name])(root=root, train=train, download=True)
    X = np.asarray(ds.data); y = np.asarray(ds.targets)
    return X.reshape(len(X), -1).astype(np.float32) / 255.0, y


def inject_noise_and_dups(X, y, noise=0.2, dup=0.3, seed=0):
    """The PDF's evaluation protocol: flip 20% of labels, add 30% extra near-copies (training set only)."""
    rng = np.random.default_rng(seed)
    X, y = X.copy(), y.copy()
    classes = np.unique(y)
    flip = rng.choice(len(y), int(noise * len(y)), replace=False)
    for i in flip:
        y[i] = rng.choice(classes[classes != y[i]])
    src = rng.choice(len(y), int(dup * len(y)), replace=False)
    Xd = X[src] + rng.normal(0, 0.01 * (X.std() + 1e-9), X[src].shape)
    return np.vstack([X, Xd]), np.concatenate([y, y[src]])   # copies inherit the (possibly flipped) label
