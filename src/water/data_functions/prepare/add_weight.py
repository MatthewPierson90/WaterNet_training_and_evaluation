import rasterio as rio
import numpy as np
from pathlib import Path
from water.basic_functions import delete_directory_contents, SharedMemoryPool

from functools import partial


def distance(row1, col1, row2, col2):
    p = 10
    return (np.abs(row1/row2 - row2/row2)**p + np.abs(col1/col2 - col2/col2)**p)**(1/p)


def weight_function(row, col, mid_row, mid_col):
    return 1/(distance(row, col, mid_row, mid_col) + 1)


def add_weight_to_raster(file_path: Path, weighted_dir: Path):
    with rio.open(file_path) as rio_f:
        data = rio_f.read()[0]
        profile = rio_f.profile
    profile['count'] += 1
    weights = np.zeros(data.shape)
    num_rows = profile['height']
    num_cols = profile['width']
    mid_row = (num_rows - 1)/2
    mid_col = (num_cols - 1)/2
    add_weight_func = np.frompyfunc(partial(weight_function, mid_row=mid_row, mid_col=mid_col), nin=2, nout=1)
    rows_cols = np.array([[row, col] for row in range(num_rows) for col in range(num_cols)], dtype=np.float32)
    rows, cols = rows_cols[:, 0], rows_cols[:, 1]
    weight_values = add_weight_func(rows, cols)
    weight_values = weight_values/weight_values.max()
    weights[rows.astype(int), cols.astype(int)] = weight_values
    data = data*weights
    weights[np.isnan(data)] = np.nan
    data = np.stack([data, weights], axis=0)
    save_path = weighted_dir/file_path.name
    with rio.open(save_path, 'w', **profile) as dst_f:
        dst_f.write(data)
    return weights


def add_weight_to_file_list(file_list, weighted_dir):
    for file in file_list:
        add_weight_to_raster(file, weighted_dir)


def add_weight_to_all_outputs(output_parent_dir, num_proc, output_name='output_data'):
    output_dir = output_parent_dir/output_name
    weighted_dir = output_parent_dir/f'{output_name}_weighted'
    if weighted_dir.exists():
        delete_directory_contents(weighted_dir)
    else:
        weighted_dir.mkdir()
    output_files = list(output_dir.iterdir())
    step_size = len(output_files)//(2*num_proc) + 1
    inputs = [{
        'file_list': output_files[i*step_size: (i+1)*step_size],
        'weighted_dir': weighted_dir
    } for i in range(2*num_proc)]
    SharedMemoryPool(num_proc=num_proc, func=add_weight_to_file_list, input_list=inputs, use_kwargs=True).run()

