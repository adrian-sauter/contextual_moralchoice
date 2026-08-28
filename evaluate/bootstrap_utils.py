"""Bootstrap confidence-interval helper shared by evaluate/ scripts and evaluate/ALL_FIGURES.ipynb."""

import numpy as np


def bootstrap_ci(data, stat_fn, B=10000, ci=(2.5, 97.5), n_samples=None):
    boot_stats = []
    n = len(data)
    if n_samples is None:
        n_samples = n
    for _ in range(B):
        sample = np.random.choice(data, size=n_samples, replace=True)
        boot_stats.append(stat_fn(sample))
    lower, upper = np.percentile(boot_stats, ci)
    return lower, upper, np.mean(boot_stats), np.std(boot_stats)
