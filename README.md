# DataDiet engine (ML engine lead)

    pip install -r requirements.txt
    python demo_digits.py                 # sklearn MLP proof run (tested)
    python demo_mnist_cnn.py --digits     # CNN smoke test, offline
    python demo_mnist_cnn.py              # real MNIST CNN (pooling x optimizer sweep)

Contract A: score_dataset, health_report, auto_select, select_subset, train_and_measure (datadiet/__init__.py)
CNN: datadiet/cnn_train.py (pool=max|avg, optimizer=adam|adamw|sgd|rmsprop), datadiet/cnn_proof.py
Note: CNN files were not executed in the build sandbox; run --digits first.

# DataDiet Benchmark Protocol — Member 2

## Overview

The completed controlled benchmark compares three training strategies:

- `full`: train on the full training split.
- `random`: train on a randomly selected subset.
- `datadiet`: train on the subset selected by DataDiet.

The completed Random Forest experiments cover three datasets, four retention fractions, and three random seeds. The consolidated `results.csv` contains 108 records.

| Dataset | Description | Records |
|---|---|---:|
| `digits` | Scikit-learn digits dataset with controlled training-set corruption | 36 |
| `mnist` | MNIST with controlled training-set corruption | 36 |
| `csv` | UCI Wine Quality red wine dataset, evaluated as supplied | 36 |
| **Total** | | **108** |

Each dataset contains 12 records for each strategy: four retention fractions multiplied by three seeds.

## Experimental protocol

Retention fractions:

- 10% (`0.1`)
- 20% (`0.2`)
- 40% (`0.4`)
- 60% (`0.6`)

Seeds: `0`, `1`, and `2`.

The controlled results use Random Forest. The separate `results_mlp_pilot.csv` file contains an exploratory, single-seed digits/MLP pilot; it is not part of the 108-record controlled Random Forest results.

### Reproduce the experiments

Install dependencies from the project requirements first.

Digits:

```bash
python -m benchmarks.run_benchmarks --dataset digits --models random_forest --fractions 0.1 0.2 0.4 0.6 --seeds 0 1 2 --output benchmarks/reproduced_digits.csv
```

MNIST (downloads MNIST through torchvision when needed):

```bash
python -m benchmarks.run_benchmarks --dataset mnist --models random_forest --fractions 0.1 0.2 0.4 0.6 --seeds 0 1 2 --output benchmarks/reproduced_mnist.csv
```

Wine Quality red:

```bash
python -m benchmarks.run_benchmarks --dataset csv --csv data/winequality_red.csv --label quality --models random_forest --fractions 0.1 0.2 0.4 0.6 --seeds 0 1 2 --output benchmarks/reproduced_wine.csv
```

These commands create separate reproduction files; they do not overwrite the consolidated `results.csv`. The runner appends results to its selected output file, so use a new filename or remove an old reproduction file before rerunning that experiment.

After preparing or changing the consolidated results file, regenerate the summary and charts with:

```bash
python -c "from benchmarks.run_benchmarks import rebuild_summary_and_plot; rebuild_summary_and_plot()"
```

This rebuild function reads `benchmarks/results.csv`.

## Dataset details

### Digits and MNIST

The benchmark loader applies label noise and near-duplicate examples to the training split only. The held-out test split remains untouched. These are controlled synthetic-corruption experiments; they should not be interpreted as direct evidence of performance on arbitrary real-world datasets.

### UCI Wine Quality — red wine

Source: UCI Machine Learning Repository, Wine Quality dataset:

https://archive.ics.uci.edu/dataset/186/wine+quality

The red wine file contains 1,599 observations, 11 numeric input features, and the `quality` target column. In the downloaded source file, the observed data checks were:

- Exact duplicate rows: 240
- Missing values: 0

The CSV benchmark uses the dataset as supplied after converting it from semicolon-separated to comma-separated format. Duplicate records were not removed. The dataset is not synthetically corrupted by the benchmark loader.

Dataset attribution: UCI Machine Learning Repository, Wine Quality dataset, by Paulo Cortez and colleagues.

## Interpreting results

- `full_seedN`, `random_seedN`, and `datadiet_seedN` identify the strategy and random seed.
- `accuracy` is test accuracy on the held-out split associated with that seed.
- The random-subset baseline averages three random-subset model fits within each seed-level run.
- `seconds` includes DataDiet scoring and selection overhead for the DataDiet strategy.
- The summary file groups results by dataset, model, retention fraction, and strategy, reporting the number of runs and mean and standard deviation of accuracy, runtime, and estimated impact.

Subset selection does not necessarily improve accuracy over training on all available data. Results should be reported as observed, including cases where the full-data model performs better or DataDiet's selection overhead outweighs its runtime savings.

## Energy, carbon, and cost

The benchmark runner now measures each strategy separately with CodeCarbon when tracking is available. CodeCarbon estimates energy use and CO2-equivalent emissions from supported hardware telemetry; it is not a physical electricity meter.

The `measurement_source` column identifies the provenance of energy/emissions:
- `codecarbon`: CodeCarbon returned both energy and emissions values.
- `estimate_fallback`: CodeCarbon was unavailable or failed, and the runtime-based estimate was used.
- `mixed`: the random-baseline row averages three trials whose sources were not all the same.

The existing `results.csv` file contains historical results from the earlier runtime-based estimate and does not have the `measurement_source` column. Do not append new rows to that older file. Use a new output file for CodeCarbon-enabled experiments, such as `benchmarks/results_codecarbon.csv`.

Install dependencies:

```bash
pip install -r requirements.txt
```

First run a small strict smoke test. `--require-codecarbon` makes the run stop with an error rather than silently using estimates; `--skip-summary` prevents the smoke test from replacing the consolidated summary or charts.

```bash
python -m benchmarks.run_benchmarks --dataset digits --models random_forest --fractions 0.1 --seeds 0 --output benchmarks/codecarbon_smoke.csv --require-codecarbon --skip-summary
```

If that completes and the output rows show `measurement_source=codecarbon`, run the full controlled sweeps into a separate file:

```bash
python -m benchmarks.run_benchmarks --dataset digits --models random_forest --fractions 0.1 0.2 0.4 0.6 --seeds 0 1 2 --output benchmarks/results_codecarbon.csv --require-codecarbon
python -m benchmarks.run_benchmarks --dataset mnist --models random_forest --fractions 0.1 0.2 0.4 0.6 --seeds 0 1 2 --output benchmarks/results_codecarbon.csv --require-codecarbon
python -m benchmarks.run_benchmarks --dataset csv --csv data/winequality_red.csv --label quality --models random_forest --fractions 0.1 0.2 0.4 0.6 --seeds 0 1 2 --output benchmarks/results_codecarbon.csv --require-codecarbon
```

These three commands append 36 rows per dataset to the new file, for 108 records total, assuming every run completes. The summary and accuracy charts are regenerated from the selected output file. Keep the original `results.csv` until the new results have passed validation and you decide which version to present.

The measurement boundaries are strategy-specific:
- `full`: full-data model fit and test scoring.
- `random`: random subset selection, model fit, and test scoring, repeated three times; runtime, energy, emissions, and cost are reported as the mean per trial.
- `datadiet`: DataDiet scoring, automatic method selection, subset selection, model fit, and test scoring. This includes the DataDiet selection overhead.

The configured rupee value is **not measured by CodeCarbon**. It remains a calculation based on elapsed time and the configured `rupees_per_hour` rate. The earlier runtime-based fallback uses these assumptions:

- Average power: 65 W
- Grid emissions factor: 0.71 kg CO2/kWh
- Illustrative compute cost: ₹25/hour

These values are configurable assumptions, not universal facts. Verify the electricity emissions factor against the latest Central Electricity Authority (CEA) CO2 Baseline Database before using figures in a formal report:

- CEA database: https://cea.nic.in/cdm-co2-baseline-database/?lang=en
- CEA database reports: https://www.cea.nic.in/cdm-co2-baseline-view?heading=CDM-CO%E2%82%82+Baseline+Database+Reports&headingHi=CDM-CO%E2%82%82+Baseline+Database+Reports&lang=en&mappingId=234

CodeCarbon documentation: https://docs.codecarbon.io/latest/tutorials/first-tracking/

CodeCarbon samples hardware power at a configurable interval (set to one second here). Very short runs may still have noisy or near-zero measurements; report the measurement method and limitations rather than treating values as exact physical readings.

## Generated artifacts

- `results.csv`: historical controlled Random Forest results using runtime-based impact estimates (108 records).
- `results_codecarbon.csv`: new CodeCarbon-enabled controlled results, once the strict rerun is completed (expected 108 records).
- `codecarbon/`: individual CodeCarbon raw report CSV files.
- `summary.csv`: grouped statistics generated from the most recently selected benchmark output file.
- `results_mlp_pilot.csv`: separate exploratory MLP pilot.
- `accuracy_digits_random_forest.png`: digits accuracy chart.
- `accuracy_mnist_random_forest.png`: MNIST accuracy chart.
- `accuracy_csv_random_forest.png`: Wine Quality accuracy chart.
- `accuracy_digits_mlp.png`: MLP pilot chart.

The measurement integration updates `datadiet/measure.py` to measure each strategy separately while preserving the public `train_and_measure` result shape.
