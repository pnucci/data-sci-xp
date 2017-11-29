import numpy as np
import pandas as pd


# def balance_oversample(x, y):
#     if len(x) != len(y):
#         raise ValueError('x and y should have same length')
#     labels, counts = np.unique(y, return_counts=True)
#     df = pd.DataFrame({'x': x, 'y': y})
#     max_count = max(counts)
#     extra_samples = []
#     for i in range(len(labels)):
#         df_filtered = df[df.y == labels[i]]
#         extra_samples_for_class = df_filtered.sample(max_count - counts[i], replace=True)
#         extra_samples.append(extra_samples_for_class)
#     balanced = df.append(extra_samples, ignore_index=True)
#     return (balanced.x.values, balanced.y.values)


def balance_oversample(x, y):
    if len(x) != len(y):
        raise ValueError('x and y should have same length')
    labels, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    extra_x = []
    extra_y = []
    for i in range(len(labels)):
        label = labels[i]
        indexes = np.argwhere(y == label).ravel()
        num_extra = max_count - counts[i]
        if (num_extra > 0):
            indexes_extra = np.random.choice(
                indexes, num_extra, replace=True)
            extra_x.extend([x[i] for i in indexes_extra])
            extra_y.extend([y[i] for i in indexes_extra])
    x2 = np.concatenate([np.array(x), extra_x])
    y2 = np.concatenate([np.array(y), extra_y])
    return (x2, y2)
