"""Step 2 - Row scores (duplicates + label confidence) and the health report."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from .embed import embed

DUP_SIM = 0.985   # cosine similarity threshold from the PDF
NOISE_P = 0.3     # label-probability threshold from the PDF
N_NEIGHBORS = 6


def find_duplicates(Z, y, thr=DUP_SIM, k=N_NEIGHBORS):
    """Flag a LATER row if an EARLIER row with the same label has cosine sim >= thr. Earlier one is kept."""
    n = len(Z)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine").fit(Z)
    dist, idx = nn.kneighbors(Z)
    sim = 1.0 - dist
    is_dup = np.zeros(n, dtype=bool)
    for i in range(n):
        for s, j in zip(sim[i], idx[i]):
            if j < i and s >= thr and y[j] == y[i]:
                is_dup[i] = True
                break
    return is_dup


def label_confidence(Z, y_idx, folds=3, seed=0):
    """Probability of the GIVEN label from 3-fold stratified CV logistic regression."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(LogisticRegression(max_iter=500), Z, y_idx, cv=cv, method="predict_proba")
    return proba[np.arange(len(y_idx)), y_idx]


def score_dataset(X, y, seed=0):
    X = np.asarray(X)
    classes, y_idx = np.unique(y, return_inverse=True)
    Z = embed(X, seed=seed)
    conf = label_confidence(Z, y_idx, seed=seed)
    return {
        "Z": Z,
        "y": y_idx,
        "is_dup": find_duplicates(Z, y_idx),
        "conf": conf,
        "loss": -np.log(np.clip(conf, 1e-12, 1.0)),   # high = hard or mislabelled
        "is_noisy": conf < NOISE_P,
    }


def health_report(scores):
    dup, noisy = scores["is_dup"], scores["is_noisy"]
    return {
        "n_rows": int(len(dup)),
        "dup_share": float(dup.mean()),
        "noisy_share": float(noisy.mean()),
        "health_score": float(100 * np.mean(~dup & ~noisy)),
    }
