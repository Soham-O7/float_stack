"""Step 1 - Embedding: turn raw feature rows into unit vectors where cosine similarity = "looks alike"."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def embed(X, max_components=32, seed=0):
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    std = X.std(axis=0)
    # soft standardisation: near-constant columns (std ~ 0) are not blown up
    Xs = (X - X.mean(axis=0)) / (std + 0.2 * std.mean() + 1e-12)
    k = max(1, min(max_components, Xs.shape[0] - 1, Xs.shape[1]))
    Z = PCA(n_components=k, random_state=seed).fit_transform(Xs)
    return normalize(Z, norm="l2")  # L2 norm -> dot product == cosine similarity
