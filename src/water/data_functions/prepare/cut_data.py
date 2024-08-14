import rasterio as rio
from rasterio.warp import Resampling
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import shapely
from water.basic_functions import SharedMemoryPool, Path, make_directory_gdf, name_to_box, delete_directory_contents
import geopandas as gpd
import numpy as np



def merge_all_in_dir(dir_path)->rxr.raster_array:
    files = dir_path.glob('*.tif')
    to_merge = [rxr.open_rasterio(file) for file in files]
    merged = merge_arrays(to_merge)
    return merged


def merge_dir_and_save(dir_path, save_dir_path, save_name='merged_data.tif'):
    merged = merge_all_in_dir(dir_path)
    merged = merged.fillna(0)
    profile = dict(
        driver='GTiff', crs=merged.rio.crs, transform=merged.rio.transform(), dtype=merged.dtype,
        width=merged.rio.width, height=merged.rio.height, count=merged.rio.count
    )
    with rio.open(save_dir_path/save_name, 'w', **profile) as rio_f:
        rio_f.write(merged.to_numpy())


def find_intersecting_files(image_path: Path, data_dir: Path, buffer: float = .05) -> list:
    try:
        im_box = name_to_box(image_path.name, buffer=buffer)
    except TypeError:
        with rio.open(image_path) as rio_f:
            im_box = shapely.box(*rio_f.bounds)
    possible_files = list(data_dir.glob('*.tif'))
    intersection_files = []
    for file in possible_files:
        box = name_to_box(file.name)
        if im_box.intersects(box):
            intersection_files.append(file)
    return intersection_files


def find_intersection_with_bbox(bbox, data_dir: Path):
    bbox = shapely.box(*bbox)
    possible_files = list(data_dir.glob('*.tif'))
    intersection_files = []
    for file in possible_files:
        box = name_to_box(file.name, buffer=0)
        if bbox.intersects(box):
            intersection_files.append(file)
    return intersection_files


def open_intersecting_rasters(files_to_merge: list,
                              target_array: rxr.raster_array,
                              extra_pixels: int) -> list:
    x_min, y_min, x_max, y_max = target_array.rio.bounds()
    arrays = []
    for file in files_to_merge:
        with rxr.open_rasterio(file) as array:
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


def mean_merge(arrays, target_array):
    no_data = np.nan
    dtype = arrays[0].dtype
    final_array = target_array.copy()
    final_array = final_array.astype(np.float32)
    final_array = final_array.rio.set_nodata(np.nan)
    arrays = [array.astype(np.float32) for array in arrays]
    arrays = np.stack([
        array.rio.set_nodata(np.nan).rio.reproject_match(
            target_array, resampling=Resampling.nearest
        ).data for array in arrays
    ])
    mean_array = np.nansum(arrays, axis=0)
    mean_array = mean_array[0:-1]/mean_array[-1:]
    final_array.data = mean_array
    final_array = final_array.rio.set_nodata(no_data)
    final_array = final_array.astype(dtype)
    return final_array


def open_intersecting_rasters_reproject_first(files_to_merge: list,
                              target_array: rxr.raster_array,
                              extra_pixels: int,
                                              resampling) -> list:
    x_min, y_min, x_max, y_max = target_array.rio.bounds()
    arrays = []
    for file in files_to_merge:
        with rxr.open_rasterio(file) as array:
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
                sub_arr = sub_arr.rio.reproject_match(target_array, resampling=resampling)
                arrays.append(sub_arr)
    return arrays


def merge_data_reproject_first(
        files_to_merge: list, target_image: Path,
        extra_pixels: int = 20, use_mean_merge: bool = False,
        resampling=Resampling.cubic
) -> rxr.raster_array:
    with rxr.open_rasterio(target_image) as target_array:
        arrays = open_intersecting_rasters_reproject_first(files_to_merge, target_array, extra_pixels, resampling)
        if len(arrays) > 0:
            if use_mean_merge:
                final_array = mean_merge(arrays, target_array)
            else:
                dtype = arrays[0].dtype
                final_array = merge_arrays(arrays)
                final_array = final_array.astype(dtype)
            return final_array
    return None


def merge_data(
        files_to_merge: list, target_image: Path,
        extra_pixels: int = 20, use_mean_merge: bool = False,
        resampling=Resampling.cubic
) -> rxr.raster_array:
    with rxr.open_rasterio(target_image) as target_array:
        arrays = open_intersecting_rasters(files_to_merge, target_array, extra_pixels)
        if len(arrays) > 0:
            if use_mean_merge:
                final_array = mean_merge(arrays, target_array)
            else:
                dtype = arrays[0].dtype
                final_array = merge_arrays(arrays)
                final_array = final_array.rio.reproject_match(target_array, resampling=resampling)
                final_array = final_array.astype(dtype)
            return final_array
    return None


def find_intersecting_files_with_data_gdf(image_path: Path, data_dir: Path, data_gdf: gpd.GeoDataFrame) -> list:
    try:
        im_box = name_to_box(image_path.name, buffer=0)
    except TypeError:
        with rio.open(image_path) as rio_f:
            im_box = shapely.box(*rio_f.bounds)
    file_names = data_gdf[data_gdf.intersects(im_box)].file_name
    intersection_files = [data_dir/file_name for file_name in file_names]
    return intersection_files


def cut_data_to_image_and_save_using_data_gdf(
        image_paths: list, data_dir: Path, data_gdf: gpd.GeoDataFrame, save_dir: Path,
        use_mean_merge: bool = False, resampling=Resampling.cubic
):
    merged_dir = save_dir
    for image_path in image_paths:
        merged_name = image_path.name
        save_path = merged_dir/merged_name
        if not save_path.exists():
            intersecting_files = find_intersecting_files_with_data_gdf(image_path, data_dir=data_dir, data_gdf=data_gdf)
            if len(intersecting_files) > 0:
                merged = merge_data(
                    intersecting_files, image_path, use_mean_merge=use_mean_merge, resampling=resampling
                )
                if merged is not None:
                    write_cut_data(image_path, save_path, merged)


def cut_data_to_image_and_save(
        save_dir: Path,
        image_paths: list, data_dir: Path,
        use_mean_merge: bool = False,
        resampling=Resampling.cubic
):
    for image_path in image_paths:
        merged_name = image_path.name
        save_path = save_dir/merged_name
        if not save_path.exists():
            intersecting_files = find_intersecting_files(image_path, data_dir=data_dir, buffer=0)
            if len(intersecting_files) > 0:
                merged = merge_data(
                    intersecting_files, image_path, use_mean_merge=use_mean_merge, resampling=resampling
                )
                if merged is not None:
                    write_cut_data(image_path, save_path, merged)


def write_cut_data(image_path: Path, save_path: Path, merged: rxr.raster_array):
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


def cut_data_to_match(
        data_dir: Path,
        match_dir: Path,
        save_dir: Path,
        num_proc: int = 12,
        use_mean_merge: bool = False,
        **kwargs
):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    else:
        delete_directory_contents(save_dir)
    file_paths = list(match_dir.glob('*'))
    print(len(file_paths))
    num_per = int(np.ceil(len(file_paths)/(num_proc*4)))
    input_list = [
        {
            'save_dir': save_dir,
            'image_paths': file_paths[i*num_per: (i + 1)*num_per],
            'data_dir': data_dir,
            'use_mean_merge': use_mean_merge,
        } for i in range(num_proc*4)
    ]
    SharedMemoryPool(
        num_proc=num_proc, func=cut_data_to_image_and_save, input_list=input_list, use_kwargs=True
    ).run()


def cut_data_to_match_file_list(
        save_dir: Path,
        data_dir: Path,
        file_paths: list[Path],
        num_proc: int = 12,
        use_mean_merge: bool = False,
        num_per_proc: int=1,
        use_file_names: bool=True,
        resampling=Resampling.cubic,
        clear_save_dir: bool = True,
        **kwargs
):

    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    elif clear_save_dir:
        delete_directory_contents(save_dir)
    data_gdf = make_directory_gdf(data_dir, use_file_names)
    num_per = int(np.ceil(len(file_paths)/(num_proc*num_per_proc)))
    input_list = [
        {
            'image_paths': file_paths[i*num_per: (i + 1)*num_per],
            'data_dir': data_dir,
            'data_gdf': data_gdf,
            'use_mean_merge': use_mean_merge,
            'save_dir': save_dir,
            'resampling': resampling
        } for i in range(num_proc*num_per_proc)
    ]
    SharedMemoryPool(
        num_proc=num_proc, func=cut_data_to_image_and_save_using_data_gdf,
        input_list=input_list, use_kwargs=True, print_progress=False
    ).run()

