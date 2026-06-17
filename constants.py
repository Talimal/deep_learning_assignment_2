SEED = 42
RESULTS_PATH = f"results_SEED={SEED}"

HYPERPARAM_GRID = {
    "lr": [1e-4, 1e-3, 1e-2],
    "margin": [1.0, 2.0],
    "batch_size": [4, 16, 32, 64]
}

TRAIN_EPOCHS = 20
