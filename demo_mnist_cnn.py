"""python demo_mnist_cnn.py            -> real MNIST (downloads via torchvision)
   python demo_mnist_cnn.py --digits   -> offline smoke test on sklearn digits (1x8x8)"""
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from datadiet import score_dataset, health_report
from datadiet.cnn_proof import cnn_proof
from datadiet.loaders import load_digits, load_torchvision, inject_noise_and_dups

if "--digits" in sys.argv:
    X, y = load_digits(); X = X / 16.0; shape = (1, 8, 8)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
else:
    Xtr, ytr = load_torchvision("MNIST", train=True)
    Xte, yte = load_torchvision("MNIST", train=False)
    shape = (1, 28, 28)
    # optional: use a 20k slice while experimenting (CPU)
    keep = np.random.default_rng(0).choice(len(ytr), 20000, replace=False); Xtr, ytr = Xtr[keep], ytr[keep]

Xtr, ytr = inject_noise_and_dups(Xtr, ytr, noise=0.2, dup=0.3)   # damage TRAIN only (test stays clean)
print("health:", health_report(score_dataset(Xtr, ytr)))
out = cnn_proof(Xtr, ytr, Xte, yte, keep_frac=0.4, shape=shape, epochs=5,
                configs=[("max", "adam"), ("avg", "adam"), ("max", "sgd")])
print("auto-select:", {k: round(v, 3) for k, v in out["method_scores"].items()}, "->", out["method"])
for r in out["table"]:
    print(f"{r['pool']:<4}{r['optimizer']:<8}{r['run']:<22} rows={r['rows']:<6} acc={r['accuracy']*100:5.1f}%  "
          f"t={r['seconds']:.1f}s  {r['gco2']:.4f} gCO2  Rs{r['rupees']:.4f}")
