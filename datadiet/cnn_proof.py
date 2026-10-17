"""Proof run with the CNN: full vs random vs DataDiet, for every (pool, optimizer) config you ask for."""
import time
import numpy as np
from .scoring import score_dataset
from .select import select_subset, auto_select
from .measure import Assumptions, account
from .cnn_train import train_cnn


def cnn_proof(Xtr, ytr, Xte, yte, keep_frac=0.4, shape=(1, 28, 28),
              configs=(("max", "adam"),), epochs=5, assumptions=Assumptions(), seed=0, auto_max_rows=10000):
    Xtr, ytr = np.asarray(Xtr), np.asarray(ytr)

    t0 = time.perf_counter()
    sc = score_dataset(Xtr, ytr, seed=seed)
    # auto-select uses a fast logistic regression; cap its rows so it stays cheap on 60k-image datasets
    sel = np.random.default_rng(seed).choice(len(ytr), min(auto_max_rows, len(ytr)), replace=False)
    method, method_acc = auto_select(Xtr[sel], ytr[sel], keep_frac, seed=seed)
    dd_idx = select_subset(sc, method, keep_frac, seed=seed)
    overhead = time.perf_counter() - t0          # counted against DataDiet -> net savings

    rows = []
    for pool, opt in configs:
        kw = dict(shape=shape, pool=pool, optimizer=opt, epochs=epochs)
        r = train_cnn(Xtr, ytr, Xte, yte, seed=seed, **kw)
        rows.append((pool, opt, "full", len(ytr), r["accuracy"], r["seconds"]))
        accs, secs = [], []
        for s in range(3):
            idx = select_subset(sc, "random", keep_frac, seed=seed + s)
            r = train_cnn(Xtr[idx], ytr[idx], Xte, yte, seed=seed + s, **kw)
            accs.append(r["accuracy"]); secs.append(r["seconds"])
        rows.append((pool, opt, "random (mean of 3)", len(idx), float(np.mean(accs)), float(np.mean(secs))))
        r = train_cnn(Xtr[dd_idx], ytr[dd_idx], Xte, yte, seed=seed, **kw)
        rows.append((pool, opt, f"datadiet ({method})", len(dd_idx), r["accuracy"], r["seconds"] + overhead))

    table = [{"pool": p, "optimizer": o, "run": n, "rows": k, "accuracy": a, "seconds": s, **account(s, assumptions)}
             for p, o, n, k, a, s in rows]
    return {"table": table, "method": method, "method_scores": method_acc, "dd_idx": dd_idx}
