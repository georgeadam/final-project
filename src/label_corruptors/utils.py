import numpy as np


def generate_multiclass_noisy_labels(y, num_classes, seed):
    label_options = np.repeat(np.arange(num_classes).reshape(1, -1), len(y), axis=0)
    sample_indices = np.arange(len(label_options))
    mask = np.ones_like(label_options).astype(bool)
    mask[sample_indices, y] = 0

    label_options = label_options[mask].reshape(len(y), num_classes - 1)

    random_state = np.random.RandomState(seed)
    indices = random_state.choice(np.arange(num_classes - 1), size=len(label_options), replace=True)

    new_labels = label_options[sample_indices, indices]

    return new_labels