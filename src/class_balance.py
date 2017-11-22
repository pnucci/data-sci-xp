import numpy as np
import pandas as pd


def balance_oversample(x, y):
    if len(x) != len(y):
        raise ValueError('x and y should have same length')
    labels, counts = np.unique(y, return_counts=True)
    df = pd.DataFrame({'x': x, 'y': y})
    max_count = max(counts)
    extra_samples = []
    for i in range(len(labels)):
        df_filtered = df[df.y == labels[i]]
        extra_samples_for_class = df_filtered.sample(max_count - counts[i], replace=True)
        extra_samples.append(extra_samples_for_class)
    balanced = df.append(extra_samples, ignore_index=True)
    return (balanced.x.values, balanced.y.values)
