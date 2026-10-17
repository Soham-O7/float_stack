import numpy as np
from sklearn.model_selection import train_test_split
from datadiet import score_dataset, health_report, train_and_measure
from datadiet.loaders import load_digits, inject_noise_and_dups

X, y = load_digits()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
Xtr, ytr = inject_noise_and_dups(Xtr, ytr)          # damage the TRAIN set only
print("health:", health_report(score_dataset(Xtr, ytr)))
out = train_and_measure(Xtr, ytr, Xte, yte, keep_frac=0.4)
print("auto-select hold-out accuracies:", {k: round(v, 3) for k, v in out["method_scores"].items()})
print("chosen:", out["method"])
for r in out["table"]:
    print(f"{r['run']:<22} rows={r['rows']:<5} acc={r['accuracy']*100:5.1f}%  t={r['seconds']:.2f}s  "
          f"{r['kwh']*1000:.4f} Wh  {r['gco2']:.4f} gCO2  Rs{r['rupees']:.5f}")
