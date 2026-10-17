"""Steps 3-4 - Subset methods and auto-select."""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .scoring import score_dataset

METHODS = ["random", "hardest", "dd_easy", "dd_hard"]


def _quotas(sizes, n_keep):
    """Each cluster contributes in proportion to its size (largest-remainder rounding)."""
    exact = n_keep * sizes / sizes.sum()
    q = np.floor(exact).astype(int)
    for c in np.argsort(-(exact - q))[: n_keep - q.sum()]:
        q[c] += 1
    return np.minimum(q, sizes)


def select_subset(scores, method, keep_frac, seed=0):
    n = len(scores["y"])
    n_keep = max(1, int(round(keep_frac * n)))
    rng = np.random.default_rng(seed)
    loss = scores["loss"]

    if method == "random":
        return np.sort(rng.choice(n, n_keep, replace=False))
    if method == "hardest":                      # no cleaning at all
        return np.sort(np.argsort(-loss)[:n_keep])
    assert method in ("dd_easy", "dd_hard"), method

    clean = np.where(~scores["is_dup"] & ~scores["is_noisy"])[0]
    chosen = []
    if len(clean) > 0:
        k = int(np.clip(n_keep // 4, 8, 200))
        k = min(k, len(clean))
        labels = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3).fit_predict(scores["Z"][clean])
        sizes = np.bincount(labels, minlength=k)
        quota = _quotas(sizes, min(n_keep, len(clean)))
        for c in range(k):
            members = clean[labels == c]
            order = np.argsort(loss[members])    # easy first
            if method == "dd_hard":
                order = order[::-1]
            chosen.extend(members[order[: quota[c]]])
    chosen = list(chosen)
    if len(chosen) < n_keep:                     # too few clean unique rows: top up
        taken = set(chosen)
        pool = [i for i in np.argsort(loss) if i not in taken]
        fill_ok = [i for i in pool if not scores["is_noisy"][i]]          # non-noisy duplicates first
        fill_bad = [i for i in pool if scores["is_noisy"][i]]
        chosen.extend((fill_ok + fill_bad)[: n_keep - len(chosen)])
    return np.sort(np.array(chosen[:n_keep]))


def _fast_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))


def auto_select(X, y, keep_frac, seed=0):
    """80/20 split; each method builds a subset from the 80%; accuracy on the 20% picks the winner."""
    X, y = np.asarray(X), np.asarray(y)
    tr, ho = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    sc = score_dataset(X[tr], y[tr], seed=seed)   # scored on the 80% only: no peeking at the hold-out
    results = {}
    for m in METHODS:
        sub = select_subset(sc, m, keep_frac, seed=seed)
        model = _fast_model().fit(X[tr][sub], y[tr][sub])
        results[m] = float(model.score(X[ho], y[ho]))
    best = max(results, key=results.get)
    return best, results
