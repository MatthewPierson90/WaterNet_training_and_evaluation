import numpy as np
from numpy.random import Generator
import torch
from torch import nn
from water.basic_functions import ppaths, tt, time_elapsed
import rasterio as rio

def open_training_data(num_to_use: int=None,
                       shuffle: bool=False,
                       condition_func: 'function'=None
                       )-> 'np.ndarray':
    """

    Parameters
    ----------
    num_to_use - int
    shuffle - bool
    condition_func - function

    Returns
    -------
    training_block - np.ndarray
    """
    training_path = ppaths.training_data_raw
    training_files = list(training_path.glob('*'))

    if shuffle:
        np.random.shuffle(training_files)

    ex1 = np.load(training_files[0])
    num_days = ex1.shape[0]
    print(ex1.shape)
    del(ex1)

    if num_to_use is None:
        num_to_use = len(training_files)*num_days
    training_block_lst = []
    current_index = 0
    for file in training_files:
        train_inst = np.load(file).astype(int)
        if condition_func is not None:
            should_use = condition_func(train_inst)
        else:
            should_use = True
        if should_use:
            for day in range(num_days):
                day_data = train_inst[day].astype(float).copy()
                day_mean = day_data.mean()
                day_data = day_data-day_mean
                day_min = day_data.min()
                day_max = day_data.max()
                day_data = 2*(day_data - day_min)/(day_max - day_min) - 1
                # print(day_data.mean())
                training_block_lst.append(day_data.astype(np.float32))
                current_index += 1
        if current_index > num_to_use:
            break
    if shuffle:
        np.random.shuffle(training_block_lst)
    training_block = np.array(training_block_lst)
    return training_block



def make_true_labels_both(scale_factor: float,
                          training_block: 'torch.tensor'=None,
                          num_to_use: int=None,
                          shuffle: bool=True,
                          condition_func: 'function'=None
                          )-> ('torch.tensor', 'torch.tensor', 'torch.tensor'):
    """

    Parameters
    ----------
    scale_factor - float
    training_block - torch.tensor
    num_to_use - int
    shuffle - bool
    condition_func - function

    Returns
    -------
    training_block_tensor, bicubic_up, labels - torch.tensor, torch.tensor, torch.tensor
    """
    if training_block is None:
        training_block = open_training_data(num_to_use=num_to_use,
                                            shuffle=shuffle,
                                            condition_func=condition_func)
    num_examples = training_block.shape[0]
    training_block_tensor = torch.tensor(training_block, dtype=float)
    bicubic_up = nn.functional.interpolate(input=training_block_tensor,
                                           scale_factor=scale_factor,
                                           mode='bicubic',
                                           align_corners=True)
    training_block_tensor = training_block_tensor.type(torch.float32)
    bicubic_up = bicubic_up.type(torch.float32)
    # print(training_block_tensor.size())
    labels = torch.ones(size=(num_examples,1))
    return training_block_tensor, bicubic_up, labels


def make_init_false_labels(scale_factor: float,
                           training_block: 'torch.tensor'=None,
                           num_to_use: int=None,
                           shuffle: bool=True,
                           condition_func: 'function'=None
                           )-> ('torch.tensor', 'torch.tensor', 'torch.tensor'):
    """

    Parameters
    ----------
    scale_factor - float
    training_block - torch.tensor
    num_to_use - int
    shuffle - bool
    condition_func - function

    Returns
    -------
    x_og, x_up, labels - torch.tensor, torch.tensor, torch.tensor
    """
    if training_block is None:
        training_block = open_training_data(num_to_use=num_to_use,
                                            shuffle=shuffle,
                                            condition_func=condition_func)
    num_examples = 3*training_block.shape[0]
    training_block_tensor = torch.tensor(training_block, dtype=float)
    nearest_up = nn.functional.interpolate(input=training_block_tensor,
                                           scale_factor=scale_factor,
                                           mode='nearest')
    linear_up = nn.functional.interpolate(input=training_block_tensor,
                                           scale_factor=scale_factor,
                                           mode='bilinear',
                                           align_corners=True)
    rolled_ex = torch.tensor(np.roll(training_block, 1, axis=0), dtype=float)
    bicubic_roll = nn.functional.interpolate(input=rolled_ex,
                                             scale_factor=scale_factor,
                                             mode='bicubic',
                                             align_corners=True)
    training_block_tensor = training_block_tensor.type(torch.float32)
    nearest_up = nearest_up.type(torch.float32)
    linear_up = linear_up.type(torch.float32)
    bicubic_roll = bicubic_roll.type(torch.float32)
    labels = torch.zeros(size=(num_examples,1))
    x_og = torch.cat([training_block_tensor, training_block_tensor, training_block_tensor], dim=0)
    x_up = torch.cat([nearest_up, linear_up, bicubic_roll], dim=0)
    return x_og, x_up, labels


def returns_true(*args
                 )-> bool:
    return True


def no_zeros(data
             )-> bool:
    if np.all(data != 0):
        return True
    else:
        return False

def augment_3d_data_func(data: 'np.ndarray'
                         )-> list('np.ndarray'):
    rot90 = np.rot90(data.copy(), axes=(1, 2))
    rot180 = np.rot90(rot90.copy(), axes=(1,2))
    rot270 = np.rot90(rot180.copy(), axes=(1, 2))
    flip = np.flip(data.copy(), axis=2)
    flip90 = np.flip(rot90.copy(), axis=2)
    flip180 = np.flip(data.copy(), axis=1)
    flip270 = np.flip(rot270.copy(), axis=2)
    return [rot90, rot180, rot270, flip, flip90, flip180, flip270]


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


def random_channel_removal(data):
    datac = data.copy()
    examples = range(datac.shape[0])
    channels = range(datac.shape[1]-1)
    random_choice = np.random.choice
    indices = np.array([(ex, random_choice(channels, 1, True)[0]) for ex in examples])
    ex = indices[:, 0]
    ch = indices[:, 1]
    datac[ex, ch] = 0
    return datac

def remove_elevation(data):
    datac = data.copy()
    datac[:, 6:-1] = 0
    return datac

def remove_sentinel(data):
    datac = data.copy()
    datac[:, :6] = np.array([np.random.choice([-1, 0, 1]) for _ in range(data.shape[0])]).reshape((-1, 1, 1, 1))
    datac[:, :6] += (np.random.random(size=(datac[:, :6].shape))-.5)/100
    return datac


class AugmentFunction:
    function_choice_counts = [0, 2, 2]
    num_iterations = 0
    def __init__(self):
        self.functions = [
            random_removal,
            remove_sentinel,
            remove_elevation
        ]
        self.function_choice_counts = AugmentFunction.function_choice_counts
        self.function_choice_weights = self._update_weights()

    def __call__(self, data):
        if AugmentFunction.num_iterations == 0:
            AugmentFunction.num_iterations += 1
            choice = 0
        else:
            choice = np.random.choice(range(len(self.functions)), p=self.function_choice_weights)
        choice = 0
        aug_names = {0: 'random_removal', 1: 'remove_sentinel', 2: 'remove_elevation'}
        self.function_choice_counts[choice] += 1
        self._update_weights()
        # print(aug_names[choice])
        # print(self.function_choice_counts)
        # print(self.function_choice_weights)
        return self.functions[choice](data)

    def _update_weights(self):
        count_total = sum(self.function_choice_counts)
        weights = [count_total-count for count in self.function_choice_counts]
        exp_weights = np.exp(weights)
        self.function_choice_weights = exp_weights/exp_weights.sum()
        # weight_func = lambda x: (count_total - x)/count_total
        # self.function_choice_weights = [weight_func(count) for count in self.function_choice_counts]
        # total_weight = sum(self.function_choice_weights)
        # self.function_choice_weights = [weight/total_weight for weight in self.function_choice_weights]
        return self.function_choice_weights

if __name__ == '__main__':
    augment_function = AugmentFunction()
    t=augment_function(np.random.random((40, 11, 100, 100)))
    t=augment_function(np.random.random((40, 11, 100, 100)))
    t=augment_function(np.random.random((40, 11, 100, 100)))
    t=augment_function(np.random.random((40, 11, 100, 100)))
    t=augment_function(np.random.random((40, 11, 100, 100)))




def augment_4d_data_func(data: 'np.ndarray'
                         ) -> np.ndarray:
    aug_funcs1 = [
        lambda x: np.rot90(x, axes=(2, 3)),
        lambda x: np.flip(x, axis=3),
        lambda x: np.flip(x, axis=2),
    ]
    # aug_funcs2 = [
    #     random_removal,
    #     remove_sentinel,
    #     remove_elevation
    # ]
    # aug_names = {0: 'random_removal', 1: 'remove_sentinel', 2: 'remove_elevation'}
    # aug_funcs = [
    #     lambda x: np.rot90(x, axes=(2, 3)), lambda x: np.flip(x, axis=3), lambda x: np.flip(x, axis=2)
    # ]
    choice_1 = np.random.choice(range(30)) % 3
    # choice_2 = np.random.choice(range(30)) % 3
    # aug_function = AugmentFunction()
    aug_data1 = aug_funcs1[choice_1](data)
    aug_data2 = random_removal(data)

    # aug_data2 = aug_function(data)
    # aug_data3 = aug_function(aug_data1)
    return np.concatenate(
            [data, aug_data1, aug_data2],
            dtype=data.dtype
    )
    # return np.concatenate(
    #         [data, rot90, rot180, flip],
    #         dtype=data.dtype
    # )
    # return np.concatenate(
    #         [data, rot90, rot180, rot270, flip, flip90, flip180, flip270, random_removed],
    #         dtype=data.dtype
    # )

# def augment_4d_data_func(data: 'np.ndarray'
#                          )-> list('np.ndarray'):
#     rot90 = np.rot90(data.copy(), axes=(2, 3))
#     rot180 = np.rot90(rot90.copy(), axes=(2, 3))
#     rot270 = np.rot90(rot180.copy(), axes=(2, 3))
#     flip = np.flip(data.copy(), axis=3)
#     flip90 = np.flip(rot90.copy(), axis=3)
#     flip180 = np.flip(data.copy(), axis=2)
#     flip270 = np.flip(rot270.copy(), axis=3)
#     return [rot90, rot180, rot270, flip, flip90, flip180, flip270]

def shuffle_data(data1: 'np.ndarray',
                 data2: 'np.ndarray'
                 ) -> ('np.ndarray', 'np.ndarray'):
    new_inds = np.arange(0, data1.shape[0])
    np.random.shuffle(new_inds)
    data1 = data1[new_inds]
    data2 = data2[new_inds]
    return data1, data2


def load_training_data(num_to_use: int=None,
                       scale1: int=1,
                       scale2: int=3,
                       condition_func_3m: 'function'=None,
                       condition_func_10m: 'function'=None,
                       shuffle: bool=True,
                       augment_data: bool=True
                       )-> ('np.ndarray', 'np.ndarray'):
    if condition_func_10m is None:
        condition_func_10m = returns_true
    if condition_func_3m is None:
        condition_func_3m = returns_true
    cut3m_path = ppaths.training_data/f'cut_up_{scale2}/cut_up_{scale1}'
    cut10m_path = ppaths.training_data/f'cut_up_{scale2}/cut_up_{scale2}'
    file_names_3m = [file_path.name for file_path in cut3m_path.glob('*')]
    file_names_10m = [file_path.name for file_path in cut10m_path.glob('*')]
    file_names = [name for name in file_names_3m if name in file_names_10m]
    del(file_names_3m)
    del(file_names_10m)
    if num_to_use is None:
        num_to_use = len(file_names)
    file_names.sort()
    np.random.shuffle(file_names)
    data_lst_3m = []
    data_lst_10m = []
    count = 0
    total_count = 0
    for file_name in file_names:
        path_3m = cut3m_path/file_name
        path_10m = cut10m_path/file_name
        with rio.open(path_3m) as src:
            data_3m = src.read()
        if condition_func_3m(data_3m):
            with rio.open(path_10m) as src:
                data_10m = src.read()
            if condition_func_10m(data_10m):
                data_3m = data_3m.astype(np.float32)
                data_3m[data_3m <= 0] = 1
                data_lst_3m.append(data_3m)
                data_10m = data_10m.astype(np.float32)
                data_10m[data_10m <= 0] = 1
                data_lst_10m.append(data_10m)
                count += 1
        if len(data_lst_3m)%int(num_to_use/20) == 0 and len(data_lst_3m)!=0:
            print(f'Completed {count} of {num_to_use}')
            if total_count<20:
                data_3m_array = np.array(data_lst_3m)
                data_10m_array = np.array(data_lst_10m)
                if augment_data:
                    data_3m_array = augment_4d_data_func(data_3m_array)
                    data_10m_array = augment_4d_data_func(data_10m_array)
                if shuffle:
                    new_inds = np.arange(0,len(data_lst_3m))
                    data_3m_array = data_3m_array[new_inds]
                    data_10m_array = data_10m_array[new_inds]
                training_sets = ppaths.training_data/f'training_sets'
                if not training_sets.exists():
                    training_sets.mkdir()
                np.save(ppaths.training_data/f'training_sets/up_{scale1}_{scale2}_{total_count}', data_3m_array)
                np.save(ppaths.training_data/f'training_sets/down_{scale1}_{scale2}_{total_count}', data_10m_array)
                data_3m_array = 0
                data_10m_array = 0
                data_lst_3m = []
                data_lst_10m = []
                total_count += 1
        if count > num_to_use:
            break
    if len(data_lst_3m) > num_to_use/30:
        data_3m_array = np.array(data_lst_3m)
        data_10m_array = np.array(data_lst_10m)
        if augment_data:
            data_3m_array = augment_4d_data_func(data_3m_array)
            data_10m_array = augment_4d_data_func(data_10m_array)
        if shuffle:
            new_inds = np.arange(0, len(data_lst_3m))
            data_3m_array = data_3m_array[new_inds]
            data_10m_array = data_10m_array[new_inds]
        np.save(ppaths.training_data/f'training_sets/up_{scale1}_{scale2}_{total_count}', data_3m_array)
        np.save(ppaths.training_data/f'training_sets/down_{scale1}_{scale2}_{total_count}', data_10m_array)
    return data_3m_array, data_10m_array




def load_naip_training_data(num_to_use: int=None,
                            condition_func_2m: 'function'=None,
                            condition_func_10m: 'function'=None,
                            shuffle: bool=True,
                            augment_data: bool=True
                            )-> ('np.ndarray', 'np.ndarray'):
    if condition_func_10m is None:
        condition_func_10m = returns_true
    if condition_func_2m is None:
        condition_func_3m = returns_true
    cut2m_path = ppaths.naip_data/'cut_up_2m'
    cut10m_path = ppaths.naip_data/'cut_up_10m'
    file_names_2m = [file_path.name for file_path in cut2m_path.glob('*')]
    file_names_10m = [file_path.name for file_path in cut10m_path.glob('*')]
    file_names = [name for name in file_names_2m if name in file_names_10m]
    del(file_names_2m)
    del(file_names_10m)
    if num_to_use is None:
        num_to_use = len(file_names)
    file_names.sort()
    data_lst_2m = []
    data_lst_10m = []
    count = 0
    for file_name in file_names:
        path_2m = cut2m_path/file_name
        path_10m = cut10m_path/file_name
        with rio.open(path_2m) as src:
            data_2m = src.read()
        if condition_func_2m(data_2m):
            with rio.open(path_10m) as src:
                data_10m = src.read()
            if condition_func_10m(data_10m):
                data_2m = data_2m.astype(np.float32)
                data_2m[data_2m==0]=1
                data_lst_2m.append(data_2m)
                data_10m = data_10m.astype(np.float32)
                data_10m[data_10m == 0] = 1
                data_lst_10m.append(data_10m)
                count += 1
        if count%1000 == 0:
            print(f'Completed {count} of {num_to_use}')
        if count > num_to_use:
            break
    data_2m_array = np.array(data_lst_2m)
    data_10m_array = np.array(data_lst_10m)
    if augment_data:
        data_2m_array = augment_4d_data_func(data_2m_array)
        data_10m_array = augment_4d_data_func(data_10m_array)
    if shuffle:
        new_inds = np.arange(0,len(data_lst_2m))
        data_2m_array = data_2m_array[new_inds]
        data_10m_array = data_10m_array[new_inds]
    return data_2m_array, data_10m_array



def augment_shuffle_tensor(data_3m,
                           data_10m,
                           dtype=torch.float32
                           )-> ('torch.tensor', 'torch.tensor', 'torch.tensor'):
    num = data_3m.shape[0]
    data_3m = data_3m.copy()
    data_10m = data_10m.copy()
    data_3m = augment_4d_data_func(data_3m)
    data_10m = augment_4d_data_func(data_10m)
    data_3m, data_10m = shuffle_data(data_3m, data_10m)
    data_3m = torch.tensor(data_3m, dtype=dtype)
    data_10m = torch.tensor(data_10m, dtype=dtype)
    ones = torch.ones(size=(num,1), dtype=dtype)
    return data_10m, data_3m, ones



def load_numpy_data(version,
                    pairs
                    ):
    up = []
    down = []
    for (x,y) in pairs:
        up_ap = np.load(ppaths.training_data/f'training_sets/up_{x}_{y}_{version}.npy')
        down_ap = np.load(ppaths.training_data/f'training_sets/down_{x}_{y}_{version}.npy')
        if len(up_ap) != 0:
            up.append(up_ap)
            down.append(down_ap)
    up = np.concatenate(up, axis=0)
    down = np.concatenate(down, axis=0)
    up, down = shuffle_data(up, down)
    return up, down

def check_same(file_name,
               path1,
               path2
               ):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2)
    with rio.open(path1/file_name) as src1:
        data1 = src1.read()
        print(src1.meta)
    with rio.open(path2/file_name) as src2:
        data2 = src2.read()
    data1 = np.transpose(data1[1:]/data1[1:].max(), (1, 2, 0))
    data2 = np.transpose(data2[1:]/data2[1:].max(), (1, 2, 0))
    print(data1.shape)
    print(data2.shape)

    ax[0].imshow(data2)
    ax[1].imshow(data1)


