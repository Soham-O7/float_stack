# DataDiet engine (ML engine lead)

    pip install -r requirements.txt
    python demo_digits.py                 # sklearn MLP proof run (tested)
    python demo_mnist_cnn.py --digits     # CNN smoke test, offline
    python demo_mnist_cnn.py              # real MNIST CNN (pooling x optimizer sweep)

Contract A: score_dataset, health_report, auto_select, select_subset, train_and_measure (datadiet/__init__.py)
CNN: datadiet/cnn_train.py (pool=max|avg, optimizer=adam|adamw|sgd|rmsprop), datadiet/cnn_proof.py
Note: CNN files were not executed in the build sandbox; run --digits first.
