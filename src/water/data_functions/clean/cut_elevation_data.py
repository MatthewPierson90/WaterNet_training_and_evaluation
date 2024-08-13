import rasterio as rio
from rasterio.warp import Resampling

import rioxarray as rxr
from rioxarray.merge import merge_arrays, merge_datasets
import shapely
from water.basic_functions import tt, time_elapsed, SharedMemoryPool
from water.basic_functions import ppaths, Path
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt


def file_name_to_bbox(file_name):
    bbox = file_name[:-4].split('_')[1:]
    bbox = [float(val) for val in bbox]
    return bbox

def name_to_box(file_name):
    bbox = file_name_to_bbox(file_name)
    return shapely.box(*bbox)

def find_intersections(image_path, elevation_dir_name='elevation'):
    im_box = name_to_box(image_path.name)
    elevation_dir = ppaths.training_data/elevation_dir_name
    # elevation_dir = Path('/media/matthew/Behemoth/data/usa_waterway_data/elevation_third')
    elevation_files = list(elevation_dir.glob('*'))
    intersection_files = []
    for elevation_file in elevation_files:
        el_box = name_to_box(elevation_file.name)
        if im_box.intersects(el_box):
            # with rio.open(elevation_file) as el_f:
            #     print(el_f.meta)
            if el_box.contains(im_box):
                return [elevation_file]
            else:
                intersection_files.append(elevation_file)
    return intersection_files


def find_intersecting_files(image_path: Path, data_dir: Path, buffer: float = .5) -> list:
    im_box = name_to_box(image_path.name, buffer=buffer)
    possible_files = list(data_dir.glob('*'))
    intersection_files = []
    for file in possible_files:
        box = name_to_box(file.name)
        if im_box.intersects(box):
            intersection_files.append(file)
    return intersection_files


def find_intersection_with_bbox(bbox, data_dir: Path):
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
        array = array.rio.set_nodata(np.nan)
        # if array.rio.nodata is None:
        #     if array.dtype in [np.float32, np.float64, np.float16]:
        #         array = array.rio.set_nodata(np.nan)
        #     else:
        #         array = array.rio.set_nodata(0)
        x_res, y_res = array.rio.resolution()
        x_res, y_res = abs(x_res), abs(y_res)
        y_extra, x_extra = extra_pixels*y_res, extra_pixels*x_res
        sub_arr = array[:,
                  (y_min - y_extra <= array.y) & (array.y <= y_max + y_extra),
                  (x_min - x_extra <= array.x) & (array.x <= x_max + x_extra)
                  ]
        if 0 not in sub_arr.shape:
            # subarray = array.rio.reproject_match(target_array, Resampling=Resampling.cubic)
            sub_arr = sub_arr.where(sub_arr < 1e30, other=array.rio.nodata)
            sub_arr = sub_arr.where(sub_arr > -999999, other=array.rio.nodata)
            sub_arr = sub_arr.rio.set_nodata(array.rio.nodata)
            # print(array.rio.nodata, sub_arr.rio.nodata)
            arrays.append(sub_arr)
            # if not np.all(subarray == np.nan):
            #     pass
    return arrays

def cut_elevation_to_image(image_paths,
                           ts='',
                           elevation_dir_name='elevation'):
    for image_path in image_paths:
        # merged_subdir = image_path.parent.name
        merged_name = image_path.name
        merged_dir = ppaths.training_data/f'{elevation_dir_name}{ts}_cut'
        save_path = merged_dir/merged_name
        if not save_path.exists():
            elevation_files = find_intersections(image_path, elevation_dir_name=elevation_dir_name)
            if len(elevation_files) > 0:
                with rxr.open_rasterio(image_path) as im_ar:
                    arrays_to_merge = open_intersecting_rasters(
                        files_to_merge=elevation_files, target_array=im_ar, extra_pixels=20
                    )
                    # print(arrays_to_merge[0].rio.nodata)
                    merged = merge_arrays(arrays_to_merge)
                    merged = merged.rio.reproject_match(im_ar, resampling=Resampling.cubic)
                # return merged
                #
                # if not (merged_dir/merged_subdir).exists():
                #     (merged_dir/merged_subdir).mkdir()
                if not np.any(np.isnan(merged.to_numpy())):
                    with rio.open(image_path) as src_f:
                        profile = src_f.meta
                        profile.update({
                            'count': 1, 'dtype': merged.dtype, 'driver': 'GTiff',
                            'crs': src_f.crs, 'nodata': merged.rio.nodata
                        })
                        # profile.update(dict(nodata=0))
                        to_write = merged.to_numpy()
                        # print(to_write.shape)
                        with rio.open(save_path, 'w', **profile) as dst_f:
                            dst_f.write(to_write)
        # merged.rio.to_raster(save_path)
        # return save_path


def cut_all_elevation(use_ts=False,
                      sen_path=None,
                      elevation_dir_name='elevation',
                      num_proc=12):
    ts = ''
    if use_ts:
        ts = '_ts'
    if sen_path is None:
        sen_path = ppaths.training_data/f'waterways{ts}_burned'
    
    if sen_path.exists():
        dir_path = ppaths.training_data/f'{elevation_dir_name}{ts}_cut'
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        file_paths = list(sen_path.glob('*'))
        print(len(file_paths))
        file_paths = [file for file in file_paths if not (dir_path/file.name).exists()]
        print(len(file_paths))
        num_per = int(np.ceil(len(file_paths)/(num_proc*4)))
        input_list = [{'image_paths': file_paths[i*num_per: (i+1)*num_per],
                       'ts': ts,
                       'elevation_dir_name': elevation_dir_name
                       }
                      for i in range(num_proc*4)]
        SharedMemoryPool(
            num_proc=num_proc, func=cut_elevation_to_image, input_list=input_list, use_kwargs=True
        ).run()


if __name__ == '__main__':
    # for ind in range(100, 2200):
    cut_all_elevation(
            use_ts=False, sen_path=ppaths.training_data/'waterways_burned',
            elevation_dir_name='elevation', num_proc=12
    )
    # cut_all_elevation(
    #         use_ts=False, sen_path=ppaths.training_data/'waterways_burned',
    #         elevation_dir_name='sentinel1_vh', num_proc=28
    # )
    # cut_all_elevation(use_ts=False, sen_path=None,
    #                   elevation_dir_name='elevation_cap',num_proc=10)




def rename_elevation_files():
    elevation_dir = ppaths.training_data/'elevation_thirds'
    # elevation_dir = Path('/media/matthew/Behemoth/data/usa_waterway_data/elevation_third')
    elevation_files = list(elevation_dir.glob('*'))
    for file in elevation_files:
        if 'USGS' in file.name:
            with rio.open(file) as rio_f:
                e, s, w, n = rio_f.bounds
            new_name = f'bbox_{e}_{s}_{w}_{n}.tif'
            new_path = file.parent/new_name
            file.rename(new_path)
    # print(min_n,max_n, min_w, max_w)
#
# if __name__ == '__main__':
#     rename_elevation_files()


def make_slope(el_data):
    x_der = np.zeros(el_data.shape)
    y_der = np.zeros(el_data.shape)


    x_del = (el_data[:, 1:] - el_data[:, :-1])/.0001
    x_der[:,1:-1] = (x_del[:, :-1] + x_del[:, 1:])/2
    x_der[:, 0] = x_del[:, 0]
    x_der[:, -1] = x_del[:, -1]

    y_del = (el_data[:-1] - el_data[1:])/.0001
    y_der[1:-1] = (y_del[:-1] + y_del[1:])/2
    y_der[0] = y_del[0]
    y_der[-1] = y_del[-1]

    grad = (x_der**2 + y_der**2)**.5
    return x_der, y_der, grad


def add_data_to_graph(ax: plt.axes, data: np.array, extent: tuple, cmap=None, ww_data: gpd.GeoSeries=None):
    ax.imshow(data, extent=extent, cmap=cmap)
    if ww_data is not None:
        ww_data.plot(ax=ax, alpha=.5)
