"""Member 2 benchmark runner.

Examples:
  python -m benchmarks.run_benchmarks --dataset digits --models mlp random_forest
  python -m benchmarks.run_benchmarks --dataset csv --csv data/messy.csv --label target
  python -m benchmarks.run_benchmarks --dataset mnist --models random_forest
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datadiet import train_and_measure
from datadiet.loaders import load_csv, load_digits, load_torchvision, inject_noise_and_dups

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results.csv"
SUMMARY = ROOT / "benchmarks" / "summary.csv"
KEEP_FRACTIONS = (0.1, 0.2, 0.4, 0.6)
RESULT_FIELDS = ["dataset", "model", "keep_frac", "run", "rows", "accuracy",
                 "seconds", "kwh", "gco2", "rupees", "measurement_source"]


def make_model(name: str, seed: int):
    if name == "mlp":
        return lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=seed),
        )
    if name == "random_forest":
        return lambda: RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1,
            min_samples_leaf=1,
        )
    raise ValueError(f"Unsupported model: {name}")


def load_dataset(name: str, csv_path: str | None, label: str | None, seed: int):
    if name == "digits":
        X, y = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=seed
        )
        X_train, y_train = inject_noise_and_dups(X_train, y_train, seed=seed)
        return X_train, X_test, y_train, y_test
    if name == "mnist":
        X_train, y_train = load_torchvision("MNIST", train=True)
        X_test, y_test = load_torchvision("MNIST", train=False)
        # Keep a reproducible stratified-sized slice so a local CPU benchmark is feasible.
        rng = np.random.default_rng(seed)
        take = min(12000, len(y_train))
        chosen = []
        for cls in np.unique(y_train):
            idx = np.flatnonzero(y_train == cls)
            chosen.extend(rng.choice(idx, max(1, round(take * len(idx) / len(y_train))),
                                     replace=False).tolist())
        chosen = np.asarray(chosen[:take])
        X_train, y_train = X_train[chosen], y_train[chosen]
        X_train, y_train = inject_noise_and_dups(X_train, y_train, seed=seed)
        return X_train, X_test, y_train, y_test
    if name == "csv":
        if not csv_path or not label:
            raise ValueError("--dataset csv requires --csv PATH and --label COLUMN")
        X, y = load_csv(csv_path, label)
        if len(X) < 20:
            raise ValueError("CSV must contain at least 20 rows for a stratified train/test split")
        if not np.isfinite(X).all():
            raise ValueError("CSV feature columns contain NaN or infinite values; clean them first")
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < 2:
            raise ValueError("CSV label column needs at least two classes, with at least 2 rows per class")
        return (*train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed),)
    raise ValueError(name)


def check_results_schema(path: Path):
    """Fail before an expensive run if an existing CSV has an incompatible schema."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    if header != RESULT_FIELDS:
        raise ValueError(
            f"Output file {path} has an older/different CSV schema. "
            "Choose a new --output path so estimate-based historical results "
            "are not silently mixed with CodeCarbon-enabled results."
        )


def append_results(rows):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    check_results_schema(RESULTS)
    exists = RESULTS.exists() and RESULTS.stat().st_size > 0
    with RESULTS.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def rebuild_summary_and_plot():
    if not RESULTS.exists():
        return
    df = pd.read_csv(RESULTS)
    # Group full/random/DataDiet rows across seeds; keep spread visible in summary.
    df["run_type"] = df["run"].str.replace(r"_seed\d+$", "", regex=True)
    group_fields = ["dataset", "model", "keep_frac", "run_type"]
    if "measurement_source" in df.columns:
        group_fields.append("measurement_source")
    grouped = df.groupby(group_fields, as_index=False).agg(
        runs=("accuracy", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        seconds_mean=("seconds", "mean"),
        kwh_mean=("kwh", "mean"),
        gco2_mean=("gco2", "mean"),
        rupees_mean=("rupees", "mean"),
    )
    grouped["accuracy_std"] = grouped["accuracy_std"].fillna(0.0)
    grouped.to_csv(SUMMARY, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for (dataset, model), part in grouped.groupby(["dataset", "model"]):
            fig, ax = plt.subplots(figsize=(8, 5))
            sources = sorted(part["measurement_source"].dropna().unique()) if "measurement_source" in part.columns else [None]
            for source in sources:
                source_part = part if source is None else part[part["measurement_source"] == source]
                suffix = f" ({source})" if source else ""
                for run_type in ("full", "random", "datadiet"):
                    series = source_part[source_part["run_type"] == run_type].sort_values("keep_frac")
                    if run_type == "full":
                        # Full-data accuracy is independent of keep fraction; show its mean as a line.
                        if not series.empty:
                            ax.axhline(series["accuracy_mean"].mean(), linestyle="--",
                                       label=f"Full data{suffix}")
                    elif not series.empty:
                        ax.errorbar(series["keep_frac"], series["accuracy_mean"],
                                    yerr=series["accuracy_std"], marker="o", capsize=3,
                                    label=f"{run_type.capitalize()}{suffix}")
            ax.set(title=f"Accuracy vs keep fraction — {dataset} / {model}",
                   xlabel="Fraction of training rows kept", ylabel="Test accuracy")
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            safe_name = f"{dataset}_{model}".replace("/", "_")
            fig.savefig(ROOT / "benchmarks" / f"accuracy_{safe_name}.png", dpi=150)
            plt.close(fig)
    except ImportError:
        print("matplotlib is not installed; CSV summaries were written but charts were skipped.")


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("digits", "mnist", "csv"), default="digits")
    parser.add_argument("--csv", dest="csv_path")
    parser.add_argument("--label")
    parser.add_argument("--models", nargs="+", choices=("mlp", "random_forest"),
                        default=["mlp", "random_forest"])
    parser.add_argument("--fractions", nargs="+", type=float, default=list(KEEP_FRACTIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", default=str(RESULTS))
    parser.add_argument(
        "--measurement-output-dir",
        default=str(ROOT / "benchmarks" / "codecarbon"),
        help="Directory for individual CodeCarbon CSV reports.",
    )
    parser.add_argument(
        "--require-codecarbon",
        action="store_true",
        help="Fail rather than falling back to assumption-based estimates.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Write benchmark rows without regenerating summary.csv or charts.",
    )
    args = parser.parse_args()

    if any(not 0 < f < 1 for f in args.fractions):
        parser.error("every keep fraction must be between 0 and 1")
    RESULTS = Path(args.output)
    if not RESULTS.is_absolute():
        RESULTS = ROOT / RESULTS
    check_results_schema(RESULTS)

    new_rows = []
    for seed in args.seeds:
        X_train, X_test, y_train, y_test = load_dataset(
            args.dataset, args.csv_path, args.label, seed
        )
        for model_name in args.models:
            model_factory = make_model(model_name, seed)
            for fraction in args.fractions:
                print(f"[{args.dataset}] model={model_name} keep={fraction:.0%} seed={seed}", flush=True)
                run_prefix = f"{args.dataset}_{model_name}_{fraction:g}_seed{seed}"
                out = train_and_measure(
                    X_train, y_train, X_test, y_test, keep_frac=fraction,
                    make_model=model_factory, seed=seed,
                    measurement_output_dir=args.measurement_output_dir,
                    require_codecarbon=args.require_codecarbon,
                    run_id_prefix=run_prefix,
                )
                for item in out["table"]:
                    run_type = ("full" if item["run"] == "full" else
                                "random" if item["run"].startswith("random") else "datadiet")
                    new_rows.append({
                        "dataset": args.dataset,
                        "model": model_name,
                        "keep_frac": fraction,
                        "run": f"{run_type}_seed{seed}",
                        "rows": item["rows"],
                        "accuracy": item["accuracy"],
                        "seconds": item["seconds"],
                        "kwh": item["kwh"],
                        "gco2": item["gco2"],
                        "rupees": item["rupees"],
                        "measurement_source": item["measurement_source"],
                    })
    append_results(new_rows)
    if not args.skip_summary:
        rebuild_summary_and_plot()
        print(f"Summary: {SUMMARY}")
    print(f"Wrote {len(new_rows)} rows to {RESULTS}")


if __name__ == "__main__":
    main()
