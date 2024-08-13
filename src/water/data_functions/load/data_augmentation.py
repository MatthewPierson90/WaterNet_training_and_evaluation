import numpy as np


def random_removal(data: np.ndarray):
    datac = data.copy()
    channels = range(datac.shape[1]-1)
    rows = range(datac.shape[2])
    cols = range(datac.shape[3])
    indices = np.array([(ch, row, col) for ch in channels for row in rows for col in cols])
    np.random.shuffle(indices)
    num_inds = int(len(channels)*len(rows)*len(cols)*.25)
    ch = indices[:num_inds, 0]
    rows = indices[:num_inds, 1]
    cols = indices[:num_inds, 2]
    means = datac.mean(axis=(2, 3))
    datac[:, ch, rows, cols] = means[:, ch]
    return datac


def augment_4d_data_func(data: 'np.ndarray'
                         ) -> np.ndarray:
    aug_funcs1 = [
        lambda x: np.rot90(x, axes=(2, 3)),
        lambda x: np.flip(x, axis=3),
        lambda x: np.flip(x, axis=2),
    ]
    choice_1 = np.random.choice(range(30)) % 3
    aug_data1 = aug_funcs1[choice_1](data)
    aug_data2 = random_removal(data)
    return np.concatenate(
            [data, aug_data1, aug_data2],
            dtype=data.dtype
    )