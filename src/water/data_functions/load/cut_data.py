import rasterio as rio
from rasterio.warp import Resampling
import xarray as xr
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import shapely
from water.basic_functions import tt, time_elapsed, SharedMemoryPool, Path, Iterable, name_to_box
from water.paths import ppaths
import numpy as np


def merge_file_list(file_list: Iterable[Path]) -> rxr.raster_array:
    to_merge = [rxr.open_rasterio(file) for file in file_list]
    merged = merge_arrays(to_merge)
    return merged


def merge_all_in_dir(dir_path: Path) -> rxr.raster_array:
    files = dir_path.glob('*')
    return merge_file_list(files)


def merge_dir_and_save(dir_path: Path, save_dir_path: Path) -> None:
    merged = merge_all_in_dir(dir_path)
    merged = merged.fillna(0)
    profile = dict(
        driver='GTiff', crs=merged.rio.crs, transform=merged.rio.transform(), dtype=merged.dtype,
        width=merged.rio.width, height=merged.rio.height, count=merged.rio.count
    )
    with rio.open(save_dir_path/'merged_data.tif', 'w', **profile) as rio_f:
        rio_f.write(merged.to_numpy())


def merge_file_list_and_save(file_list: list[Path], save_dir_path: Path, save_name: str = 'merged_data.tif') -> None:
    merged = merge_file_list(file_list)
    merged = merged.fillna(0)
    merged = merged.where(merged != 0, other=np.nan)
    profile = dict(
        driver='GTiff', crs=merged.rio.crs, transform=merged.rio.transform(), dtype=merged.dtype,
        width=merged.rio.width, height=merged.rio.height, count=merged.rio.count, nodata=np.nan
    )
    with rio.open(save_dir_path/save_name, 'w', **profile) as rio_f:
        rio_f.write(merged.to_numpy())


def find_intersecting_files(image_path: Path, data_dir: Path, buffer: float = .5) -> list[Path]:
    im_box = name_to_box(image_path.name, buffer=buffer)
    possible_files = list(data_dir.glob('*'))
    intersection_files = []
    for file in possible_files:
        box = name_to_box(file.name)
        if im_box.intersects(box):
            intersection_files.append(file)
    return intersection_files


def find_intersection_with_bbox(bbox: tuple[float], data_dir: Path) -> list[Path]:
    box = shapely.box(*bbox)
    possible_files = list(data_dir.glob('*'))
    intersection_files = []
    for file in possible_files:
        box = name_to_box(file.name)
        if box.intersects(box):
            intersection_files.append(file)
    return intersection_files


def open_intersecting_rasters(files_to_merge: list,
                              target_array: rxr.raster_array,
                              extra_pixels: int) -> list:
    x_min, y_min, x_max, y_max = target_array.rio.bounds()
    arrays = []
    for file in files_to_merge:
        array = rxr.open_rasterio(file)
        if array.rio.nodata is None:
            if array.dtype in [np.float32, np.float64, np.float16]:
                array = array.rio.set_nodata(np.nan)
            else:
                array = array.rio.set_nodata(0)
        x_res, y_res = array.rio.resolution()
        x_res, y_res = abs(x_res), abs(y_res)
        y_extra, x_extra = extra_pixels*y_res, extra_pixels*x_res
        sub_arr = array[:,
                  (y_min - y_extra <= array.y) & (array.y <= y_max + y_extra),
                  (x_min - x_extra <= array.x) & (array.x <= x_max + x_extra)
                  ]
        if 0 not in sub_arr.shape:
            sub_arr = sub_arr.where(sub_arr < 1e30, other=array.rio.nodata)
            arrays.append(sub_arr)
    return arrays


def mean_merge(arrays: list[rxr.raster_array], target_array: rxr.raster_array) -> rxr.raster_array:
    no_data = 0
    dtype = arrays[0].dtype
    final_array = target_array.copy()
    final_array = final_array.astype(np.float32)
    final_array = final_array.rio.set_nodata(np.nan)
    arrays = [array.astype(np.float32) for array in arrays]
    arrays = [
        array.rio.set_nodata(np.nan).rio.reproject_match(target_array, resampling=Resampling.cubic) for array in arrays
    ]
    dims = ['ind'] + list(arrays[0].dims)
    arrays = xr.DataArray(arrays, dims=dims)
    mean_array = arrays.mean(dim='ind', skipna=True)
    final_array.data = mean_array.data
    final_array = final_array.rio.set_nodata(no_data)
    final_array = final_array.astype(dtype)
    return final_array


def reproject_match_data(
        files: Path, target_image: Path,
) -> rxr.raster_array:
    with rxr.open_rasterio(target_image) as target_array:
        array = rxr.open_rasterio(files)
        dtype = array.dtype
        final_array = array.rio.reproject_match(target_array, resampling=Resampling.cubic)
        final_array = final_array.astype(dtype)
    return final_array


def cut_data_to_image_and_save(
        image_paths: list, data_dir: Path,
        base_dir_path: Path = ppaths.training_data/f'country_data',
) -> None:
    for image_path in image_paths:
        image_path = image_path/'waterways_burned.tif'
        save_dir = base_dir_path/image_path.parent.name
        if not save_dir.exists():
            save_dir.mkdir()
        save_name = f'{data_dir.name}.tif'
        save_path = save_dir/save_name
        if not save_path.exists():
            file_name = image_path.parent.parent.name
            intersecting_file = data_dir/f'{file_name}.tif'
            array = reproject_match_data(intersecting_file, image_path)
            write_cut_data(image_path, save_path, array)


def write_cut_data(image_path: Path, save_path: Path, merged: rxr.raster_array) -> None:
    with rio.open(image_path) as src_f:
        profile = src_f.meta
        profile.update(
                {
                    'count': merged.shape[0],
                    'dtype': merged.dtype,
                    'driver': 'GTiff',
                    'crs': src_f.crs,
                }
        )
        to_write = merged.to_numpy()
        with rio.open(save_path, 'w', **profile) as dst_f:
            dst_f.write(to_write)


def cut_data_to_match_file_list(
        data_dir: Path,
        file_paths: list[Path],
        num_proc: int = 12,
        base_dir_path: dir = ppaths.training_data/f'country_data/',
        **kwargs
):
    if not base_dir_path.exists():
        base_dir_path.mkdir(parents=True)
    num_per = int(np.ceil(len(file_paths)/(num_proc)))
    input_list = [
        {
            'image_paths': file_paths[i*num_per: (i + 1)*num_per],
            'data_dir': data_dir,
            'base_dir_path': base_dir_path
        } for i in range(num_proc)
    ]
    SharedMemoryPool(
            num_proc=num_proc, func=cut_data_to_image_and_save,
            input_list=input_list, use_kwargs=True, print_progress=False
    ).run()

if __name__ == '__main__':
    all_files = []
    data_dir = ppaths.training_data/'model_inputs_208'
    init_dir = data_dir/'input_data'
    base_dirs = list(init_dir.glob('*'))[:1000]
    for sub_dir in base_dirs:
        all_files.extend(sub_dir.iterdir())
    # print(all_files)
    s = tt()
    cut_data_to_match_file_list(
        data_dir=ppaths.training_data/'sentinel', file_paths=all_files[:1000], base_dir_path=data_dir, num_proc=15
    )
    cut_data_to_match_file_list(
        data_dir=ppaths.training_data/'elevation_cut', file_paths=all_files[:1000], base_dir_path=data_dir, num_proc=15
    )
    time_elapsed(s)
