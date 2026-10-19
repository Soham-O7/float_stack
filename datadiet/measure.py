"""Proof run and per-strategy energy / carbon / cost accounting."""
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .scoring import score_dataset
from .select import select_subset, auto_select


@dataclass
class Assumptions:
    power_w: float = 65.0
    grid_kg_per_kwh: float = 0.71
    rupees_per_hour: float = 25.0


def account(seconds, a: Assumptions):
    """Estimate impact from configured power, grid-intensity, and cost assumptions."""
    kwh = a.power_w * seconds / 3.6e6
    return {
        "kwh": kwh,
        "gco2": kwh * a.grid_kg_per_kwh * 1000,
        "rupees": a.rupees_per_hour * seconds / 3600,
    }


def default_model():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=0),
    )


def train_and_measure(
    X_train,
    y_train,
    X_test,
    y_test,
    keep_frac,
    make_model=default_model,
    assumptions=None,
    seed=0,
    measurement_output_dir="benchmarks/codecarbon",
    require_codecarbon=False,
    run_id_prefix="benchmark",
):
    """Compare full, random-subset, and DataDiet strategies.

    Each strategy is measured independently. DataDiet's scoring, auto-selection,
    subset selection, and final fit are all inside its measurement boundary. The
    random baseline includes subset selection and averages three independently
    measured random trials. If CodeCarbon fails, results are marked as estimates
    unless ``require_codecarbon`` is true.
    """
    # Import locally to avoid a module cycle: measurement.py imports Assumptions/account.
    from measurement import measure_callable

    assumptions = assumptions or Assumptions()
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    n_keep = max(1, int(round(keep_frac * len(y_train))))
    allow_fallback = not require_codecarbon

    def measure(fn, suffix):
        return measure_callable(
            fn,
            assumptions=assumptions,
            project_name="float_stack",
            output_dir=measurement_output_dir,
            country_iso_code="IND",
            run_id=f"{run_id_prefix}_{suffix}",
            allow_fallback=allow_fallback,
        )

    # Full-data baseline: train and evaluate only on the full training split.
    full_accuracy, full_measurement = measure(
        lambda: float(make_model().fit(X_train, y_train).score(X_test, y_test)),
        "full",
    )

    # Random baseline: include subset selection in each measured trial.
    random_accuracies = []
    random_seconds = []
    random_values = []
    random_sources = []
    for repeat in range(3):
        def run_random(repeat=repeat):
            rng = np.random.default_rng(seed + repeat)
            idx = np.sort(rng.choice(len(y_train), n_keep, replace=False))
            fitted = make_model().fit(X_train[idx], y_train[idx])
            return float(fitted.score(X_test, y_test))

        accuracy, measurement = measure(run_random, f"random_trial{repeat + 1}")
        random_accuracies.append(accuracy)
        random_seconds.append(measurement.seconds)
        random_values.append(measurement.values)
        random_sources.append(measurement.source)

    # DataDiet strategy: all scoring and selection overhead is included in its result.
    def run_datadiet():
        scores = score_dataset(X_train, y_train, seed=seed)
        method, method_scores = auto_select(X_train, y_train, keep_frac, seed=seed)
        dd_idx = select_subset(scores, method, keep_frac, seed=seed)
        fitted = make_model().fit(X_train[dd_idx], y_train[dd_idx])
        accuracy = float(fitted.score(X_test, y_test))
        return accuracy, len(dd_idx), method, method_scores

    dd_result, dd_measurement = measure(run_datadiet, "datadiet")
    dd_accuracy, dd_rows, method, method_scores = dd_result

    def mean_metric(key):
        return float(np.mean([values[key] for values in random_values]))

    random_source_set = set(random_sources)
    random_source = (
        next(iter(random_source_set)) if len(random_source_set) == 1 else "mixed"
    )

    table = [
        {
            "run": "full",
            "rows": len(y_train),
            "accuracy": full_accuracy,
            "seconds": full_measurement.seconds,
            **full_measurement.values,
            "measurement_source": full_measurement.source,
        },
        {
            "run": "random (mean of 3)",
            "rows": n_keep,
            "accuracy": float(np.mean(random_accuracies)),
            "seconds": float(np.mean(random_seconds)),
            "kwh": mean_metric("kwh"),
            "gco2": mean_metric("gco2"),
            "rupees": mean_metric("rupees"),
            "measurement_source": random_source,
        },
        {
            "run": f"datadiet ({method})",
            "rows": dd_rows,
            "accuracy": dd_accuracy,
            "seconds": dd_measurement.seconds,
            **dd_measurement.values,
            "measurement_source": dd_measurement.source,
        },
    ]
    return {
        "table": table,
        "method": method,
        "method_scores": method_scores,
    }
