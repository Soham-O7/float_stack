"""Step 5 - Proof run and energy / carbon / cost accounting."""
import time
from dataclasses import dataclass
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .scoring import score_dataset
from .select import select_subset, auto_select


@dataclass
class Assumptions:
    power_w: float = 65.0       # average machine power draw (W) - editable
    grid_kg_per_kwh: float = 0.71   # grid carbon intensity (kg CO2 / kWh) - editable; check your region
    rupees_per_hour: float = 25.0   # compute price (INR/h) - editable


def account(seconds, a: Assumptions):
    kwh = a.power_w * seconds / 3.6e6
    return {"kwh": kwh, "gco2": kwh * a.grid_kg_per_kwh * 1000, "rupees": a.rupees_per_hour * seconds / 3600}


def default_model():
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=0))


def _run(make_model, X, y, Xte, yte, extra_seconds=0.0):
    t0 = time.perf_counter()
    m = make_model().fit(X, y)
    secs = time.perf_counter() - t0 + extra_seconds
    return float(m.score(Xte, yte)), secs


def train_and_measure(X_train, y_train, X_test, y_test, keep_frac, make_model=default_model,
                      assumptions=Assumptions(), seed=0):
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)

    # DataDiet cost (scoring + auto-select + selection) is counted against it -> savings are NET
    t0 = time.perf_counter()
    sc = score_dataset(X_train, y_train, seed=seed)
    method, method_acc = auto_select(X_train, y_train, keep_frac, seed=seed)
    dd_idx = select_subset(sc, method, keep_frac, seed=seed)
    overhead = time.perf_counter() - t0

    rows = []
    acc, s = _run(make_model, X_train, y_train, X_test, y_test)
    rows.append(("full", len(y_train), acc, s))

    accs, secs = [], []
    for r in range(3):                            # random baseline = mean of 3 seeds
        idx = select_subset(sc, "random", keep_frac, seed=seed + r)
        a_, s_ = _run(make_model, X_train[idx], y_train[idx], X_test, y_test)
        accs.append(a_); secs.append(s_)
    rows.append(("random (mean of 3)", len(idx), float(np.mean(accs)), float(np.mean(secs))))

    acc, s = _run(make_model, X_train[dd_idx], y_train[dd_idx], X_test, y_test, extra_seconds=overhead)
    rows.append((f"datadiet ({method})", len(dd_idx), acc, s))

    table = [{"run": n, "rows": r, "accuracy": a, "seconds": s, **account(s, assumptions)} for n, r, a, s in rows]
    return {"table": table, "method": method, "method_scores": method_acc, "dd_idx": dd_idx}
