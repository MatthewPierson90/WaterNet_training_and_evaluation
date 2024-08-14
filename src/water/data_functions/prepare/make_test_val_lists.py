import os
import shutil

import shapely

from water.basic_functions import ppaths, get_hu4_hull_polygon, name_to_box, tt, time_elapsed
import numpy as np
from pathlib import Path


def get_dir_paths(data_path:Path):
    input_paths = data_path / 'input_data'
    val_paths = data_path / 'val_data'
    test_paths = data_path / 'test_data'
    if not val_paths.exists():
        val_paths.mkdir()
    if not test_paths.exists():
        test_paths.mkdir()
    return data_path, input_paths, val_paths, test_paths


def make_val_file_list():
    file_names = [file.name for file in (ppaths.model_inputs_832/'input_data').iterdir()]
    np.random.shuffle(file_names)
    with open(ppaths.training_data/'val_file_list.txt', 'w') as f:
        for file_name in file_names[:5000]:
            f.write(f'{file_name}\n')
    return ppaths.training_data/'val_file_list.txt'


def move_files(file_list, new_dir, old_dir):
    if not new_dir.exists():
        new_dir.mkdir()
    for file_name in file_list:
        if (old_dir/file_name).exists():
            old_path = old_dir/file_name
            new_path = new_dir/file_name
            if new_path.exists():
                shutil.copytree(old_path, new_path, dirs_exist_ok=True)
                shutil.rmtree(old_path)
            else:
                old_path.rename(new_path)
            # if len(list(old_path.iterdir())) == 0:
            #     os.remove(old_path)


def move_files_in_file_list_file(file_list_file: Path, new_dir: Path, old_dir: Path):
    with open(file_list_file, 'r') as f:
        file_list = f.read().split('\n')[:-1]
    move_files(file_list, new_dir=new_dir, old_dir=old_dir)


def select_random_hu4_indices():
    index_list = []
    dir_path = ppaths.hu4_parquet
    # Choose one hu4 index per hu2 index
    for i in range(100, 1900, 100):
        print(i)
        keep_going = True
        while keep_going:
            random_int = np.random.randint(low=i, high=i+30)
            if (dir_path / f'hu4_model_{random_int:04d}.parquet').exists():
                index_list.append(random_int)
                break
    return index_list


def make_file_bboxes(file_names: np.ndarray):
    file_name_to_bbox = lambda x: name_to_box(x, 0)
    file_name_to_bbox_ufunc = np.frompyfunc(file_name_to_bbox, 1, 1)
    bboxes = file_name_to_bbox_ufunc(file_names)
    return bboxes


def find_files_intersecting_index_list(index_list: list, file_dir: Path) -> list:
    file_paths = list(file_dir.glob('bbox*'))
    bboxes = make_file_bboxes(np.array([file.name for file in file_paths]))
    str_tree = shapely.STRtree(bboxes)
    intersecting_file_indices = []
    for hu4_index in index_list:
        hu4_geometry = get_hu4_hull_polygon(hu4_index)
        intersecting_file_indices.extend(str_tree.query(hu4_geometry, predicate='intersects'))
    intersecting_file_indices = np.unique(intersecting_file_indices)
    intersecting_files = [file_paths[index] for index in intersecting_file_indices]
    return intersecting_files


def move_hu4_test_and_val_data(hu4_index_list=None):
    if hu4_index_list is not None:
        hu4_index_list = select_random_hu4_indices()
    with open(ppaths.training_data/'test_hu4_indices.txt', 'w') as f:
        f.write('hu4_indices\n')
        for hu4_index in hu4_index_list:
            f.write(f'{hu4_index}\n')
    file_paths = find_files_intersecting_index_list(
        index_list=hu4_index_list, file_dir=ppaths.model_inputs_832/'input_data'
    )
    print(len(file_paths))
    test_file = ppaths.training_data/'test_file_list.txt'
    with open(test_file, 'w') as f:
        for file_path in file_paths:
            f.write(f'{file_path.name}\n')

    _, input_data, val_data, test_data = get_dir_paths(ppaths.model_inputs_832)
    move_files_in_file_list_file(test_file, test_data, input_data)
    val_file = make_val_file_list()
    move_files_in_file_list_file(val_file, val_data, input_data)


    _, input_data, val_data, test_data = get_dir_paths(ppaths.model_inputs_224)
    move_files_in_file_list_file(test_file, test_data, input_data)
    move_files_in_file_list_file(val_file, val_data, input_data)
