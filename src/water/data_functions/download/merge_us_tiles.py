import os

import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
import pandas as pd
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import rasterio as rio
from water.basic_functions import ppaths, my_pool, tt, time_elapsed, printdf, delete_directory_contents
import numpy as np
from pathlib import Path
from time import time
import shutil

def sort_array_list(array_list: list[rxr.raster_array]):
    array_list.sort(key=lambda x: (x == 0).sum().values)
    return array_list


# def edit_scl_data(scl_data: np.ndarray, num_steps):
#     scl_data = scl_data[:, 0]
#     scl_data[scl_data > 0] = 1
#     num_time, num_rows, num_cols = scl_data.shape
#     bad_time, bad_rows, bad_cols = np.where(scl_data == 0)
#     changed_data = scl_data.copy()
#     changed_data[bad_time, bad_rows, bad_cols] = 0
#     for i in range(-num_steps, num_steps+1):
#         rows = bad_rows - i
#         for j in range(-num_steps, num_steps+1):
#             if i != 0 or j != 0:
#                 cols = bad_cols - j
#                 inds = np.where((0 <= rows) & (rows < num_rows) & (0 <= cols) & (cols < num_cols))[0]
#                 rows_inds = rows[inds]
#                 cols = cols[inds]
#                 time = bad_time[inds]
#                 changed_data[time, rows_inds, cols] = 0
#     new_sum = changed_data.sum(axis=0)
#     good_rows, good_cols = np.where(new_sum > 0)
#     scl_data[:, good_rows, good_cols] = changed_data[:, good_rows, good_cols]
#     new_sum = scl_data.sum(axis=0)
#     bad_rows, bad_cols = np.where(new_sum == 0)
#     scl_data[:, bad_rows, bad_cols] = 1
#     scl_data = np.stack([scl_data], axis=1)
#     return scl_data


def embed_plus_one(array: np.ndarray):
    array = np.moveaxis(array, (-2, -1), (0, 1))
    num_rows, num_cols, *other = array.shape
    new_array = np.zeros((num_rows+2, num_cols+2, *other), dtype=array.dtype)
    new_array[1:-1, 1:-1] = array
    new_array = np.moveaxis(new_array, (0, 1), (-2, -1))
    array = np.moveaxis(array, (0, 1), (-2, -1))

    return new_array


def edit_scl_data(scl_data: np.ndarray, num_steps):
    scl_data = scl_data[:, 0]
    scl_data[scl_data > 0] = 1
    # s = tt()
    embeded_data = embed_plus_one(scl_data)
    embeded_data[:, :-2, 1:-1] += scl_data
    embeded_data[:, 2:, 1:-1] += scl_data
    embeded_data[:, 1:-1, :-2] += scl_data
    embeded_data[:, 1:-1, 2:] += scl_data
    embeded_data = embeded_data[:, 1:-1, 1:-1]
    num_time, num_rows, num_cols = scl_data.shape
    bad_time, bad_rows, bad_cols = np.where((embeded_data > 0) & (scl_data == 0))
    changed_data = scl_data.copy()
    changed_data[bad_time, bad_rows, bad_cols] = 0
    # time_elapsed(s, 4)
    # s = tt()
    for i in range(len(changed_data)):
        bad_inds = np.where(bad_time == i)
        rows, cols = bad_rows[bad_inds], bad_cols[bad_inds]
        for row, col in zip(rows, cols):
            row_min = max(row - num_steps, 0)
            row_max = min(row + num_steps + 1, num_rows)
            col_min = max(col - num_steps, 0)
            col_max = min(col + num_steps + 1, num_cols)
            changed_data[i, row_min: row_max, col_min: col_max] = 0
    # time_elapsed(s, 4)
    # s = tt()
    new_sum = changed_data.sum(axis=0)
    good_rows, good_cols = np.where(new_sum > 0)
    scl_data[:, good_rows, good_cols] = changed_data[:, good_rows, good_cols]
    new_sum = scl_data.sum(axis=0)
    bad_rows, bad_cols = np.where(new_sum == 0)
    scl_data[:, bad_rows, bad_cols] = 1
    scl_data = np.stack([scl_data], axis=1)
    # time_elapsed(s, 4)
    return scl_data


def mean_merge(array_list: list[rxr.raster_array], scl_list: list[rxr.raster_array], num_steps):
    scl_data = np.stack([scl.where(scl.isin([2, 4, 5, 6, 7, 11]), other=0) for scl in scl_list], axis=0)
    scl_data = edit_scl_data(scl_data, num_steps)
    data = np.stack([array.to_numpy().astype(np.float16) for array in array_list], axis=0)
    data = data*scl_data.astype(np.float16)
    data[data == 0] = np.nan
    data = np.nan_to_num(np.nanmean(data, axis=0)).astype(np.uint8)
    array = array_list[0]
    array.data = data
    return array


def sort_and_merge_array_list(array_list: list[rxr.raster_array], scl_list: list[rxr.raster_array], num_steps):
    # array_list = sort_array_list(array_list)
    # merged = merge_arrays(array_list)
    merged = mean_merge(array_list, scl_list, num_steps)

    return merged


def save_merged_array(dst_path: Path, src_file_path: Path, array: rxr.raster_array):
    with rio.open(src_file_path) as src_f:
        profile = src_f.profile
    with rio.open(dst_path, 'w', **profile) as dst_f:
        dst_f.write(array)


def get_file_list(base_path: Path):
    image_files = [
        file for file in base_path.glob('*.tif') if '_scl.tif' not in file.name
    ]
    scl_files = [
        file for file in base_path.glob('*_scl.tif')
    ]

    image_files.sort(key=lambda x: x.name)
    scl_files.sort(key=lambda x: x.name)
    return image_files, scl_files


def get_array_list(file_list: list[Path]):
    return [rxr.open_rasterio(file) for file in file_list]


def remove_old(base_path: Path):
    for file in base_path.iterdir():
        # if '_scl.tif' not in file.name:
        os.remove(file)


def merge_save_and_remove(base_path: Path, save_dir: Path, num_steps=0):
    """
    Merge, save, and delete downloaded sentinel tiles. The goal is to obtain mostly cloud free images. num_steps
    is the number of grid cells near a cell marked cloud in the scl file that the algorithm will ignore
    assuming there is at least one image with a grid cell marked as cloud free at that location. This is done because
    the scl files typically don't label an entire cloud as a cloud.

    Parameters
    ----------
    base_path - path to the base directory
    save_dir - path to the directory to save the merged rasters
    num_steps - as described above.
    """
    image_files, scl_files = get_file_list(base_path)
    array_list = get_array_list(image_files)
    scl_list = get_array_list(scl_files)
    if len(array_list) > 1:
        merged_array = sort_and_merge_array_list(array_list, scl_list, num_steps)
    else:
        merged_array = array_list[0]
    save_as = save_dir/f'{base_path.name}.tif'
    save_merged_array(
        dst_path=save_as, src_file_path=image_files[0], array=merged_array
    )
    save_storage = ppaths.waterway/'waterway_storage/sentinel_4326'
    os.system(f'cp {save_as} {save_storage}')
    remove_old(base_path)
    # os.remove(save_as)


def merge_save_and_remove_multi(base_dir: Path, save_dir: Path, num_proc: int, num_steps: int):
    inputs = [
        dict(base_path=base_path, save_dir=save_dir, num_steps=num_steps)
        for base_path in list(base_dir.iterdir())
        if len(get_file_list(base_path)[0]) > 0 and not (save_dir/f'{base_path.name}.tif').exists()
    ]
    print(len(inputs))
    if not save_dir.exists():
        save_dir.mkdir()
    my_pool(func=merge_save_and_remove, input_list=inputs, use_kwargs=True, num_proc=num_proc)


def copy_merge_save_and_remove(src_dir: Path, temp_dir: Path, save_dir: Path, num_steps: int):
    # os.system(f'cp -r {src_dir} {temp_dir}')
    shutil.copytree(src_dir, temp_dir, symlinks=True)
    merge_save_and_remove(base_path=temp_dir/src_dir.name, save_dir=save_dir, num_steps=num_steps)

def check_merged_file(merged_dir: Path):
    data_info = {'file_name': [], 'num_rows': [], 'num_cols': [], 'total': [], 'num_zero': [], 'zero_percent': []}
    file_list = list(merged_dir.iterdir())
    num_files = len(file_list)
    s = tt()
    for ind, file in enumerate(file_list):
        with rio.open(file) as rio_f:
            data = rio_f.read()
            num_rows = data.shape[-2]
            num_cols = data.shape[-1]
            total = num_rows*num_cols
            data = data.astype(np.int16).sum(axis=0)
            zero_rows, _ = np.where(data == 0)
            data_info['file_name'].append(file.name)
            data_info['num_zero'].append(len(zero_rows))
            data_info['num_rows'].append(num_rows)
            data_info['num_cols'].append(num_cols)
            data_info['total'].append(total)
            data_info['zero_percent'].append(len(zero_rows)/total)
        if ind % max(num_files//10, 1) == 0:
            print(f'Completed {(ind+1)/num_files:%} ({ind+1}/{num_files})')
            time_elapsed(s, 2)
    return data_info


def remove_bad_files(tile_dir, merged_dir, df, min_missing=.1):
    file_names = df[df.zero_percent > min_missing].file_name
    for file_name in file_names:
        file_name_split = file_name.split('.tif')[0]
        if (merged_dir/file_name).exists():
            os.remove(merged_dir/file_name)
        if (tile_dir/file_name_split).exists():
            delete_directory_contents(tile_dir/file_name_split)


def remove_europe(tile_dirs: Path, merged_path: Path):
    count = 0
    for tile in tile_dirs.iterdir():
        tile_name = tile.name
        lat = float(tile_name.split('_')[2])
        if lat > 30:
            # print(tile_name)
            # count+=1
            delete_directory_contents(tile)
            tile.rmdir()
    for file in merged_path.iterdir():
        file_name = file.name
        lat = float(file_name.split('_')[2])
        if lat > 30:
            os.remove(file)
    print(count)


def check_storage_files(base_dir: Path):
    for subdir in base_dir.iterdir():
        for file in subdir.iterdir():
            days_since = (time() - file.stat().st_mtime)/(60*60*24)
            if 1 < days_since:
                parent = file.parent.name
                sen_tile = ppaths.country_data/f'sentinel_tiles/{parent}'
                os.remove(file)
                # if sen_tile.exists():
                #     delete_directory_contents(sen_tile)
                # else:
                #     delete_directory_contents(subdir)
                #     subdir.rmdir()
                #     break
        # break


def check_scl_files(base_dir: Path):
    storage_dir = ppaths.waterway/f'waterway_storage/{base_dir.name}'
    for subdir in base_dir.iterdir():
        storage_subdir = storage_dir/subdir.name
        for file in subdir.glob('*_scl.tif'):
            storage_scl_file = storage_subdir/file.name
            image_name = file.name.replace('_scl.tif', '.tif')
            image_path = subdir/image_name
            storage_image_file = storage_subdir/image_name
            if not storage_image_file.exists() and storage_scl_file.exists():
                os.remove(storage_scl_file)
                os.remove(file)
                print(file)
                # if storage_file.exists():
                #     os.remove(storage_file)



if __name__ == '__main__':
    # check_scl_files(ppaths.country_data/'sentinel_tiles')
    # check_storage_files(ppaths.waterway/'waterway_storage/sentinel_tiles')

    # bp = ppaths.country_data/'sentinel_tiles_africa'
    # sd = ppaths.country_data/'sentinel_4326'
    #
    # merge_save_and_remove_multi(base_dir=bp, save_dir=sd, num_proc=4, num_steps=500)

    # bp = ppaths.country_data/'sentinel_tiles_europe'
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    sd = ppaths.country_data/'sentinel_4326'
    bp = ppaths.country_data/'sentinel_tiles_africa'
    # sd = ppaths.waterway/'waterway_storage/sentinel_4326'
    merge_save_and_remove_multi(base_dir=bp, save_dir=sd, num_proc=4, num_steps=500)

    # sd = ppaths.country_data/'sentinel_4326'
    # bp = ppaths.country_data/'sentinel_tiles_europe'
    # # sd = ppaths.waterway/'waterway_storage/sentinel_4326'
    # merge_save_and_remove_multi(base_dir=bp, save_dir=sd, num_proc=4, num_steps=500)

    # bp = ppaths.country_data/'sentinel_tiles_usa'
    # sd = ppaths.country_data/'sentinel_4326'
    # merge_save_and_remove_multi(base_dir=bp, save_dir=sd, num_proc=6, num_steps=500)


    # bp = ppaths.country_data/'sentinel_tiles_europe'
    # sd = ppaths.country_data/'sentinel_4326_europe'
    # merge_save_and_remove_multi(base_dir=bp, save_dir=sd, num_proc=4, num_steps=500)
    #
    # bp = ppaths.country_data/'sentinel_tiles_south_america'
    # sd = ppaths.country_data/'sentinel_4326_south_america'
    # merge_save_and_remove_multi(base_dir=bp, save_dir=sd, num_proc=4, num_steps=500)
#     # remove_europe(bp, sd)
# #     di = check_merged_file(sd)
# #     df = pd.DataFrame(di)
# #     df = df.sort_values(by='zero_percent', ascending=False)
# #     printdf(df, 100)
