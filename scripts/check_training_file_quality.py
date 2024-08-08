import pandas as pd
import geopandas as gpd
from water.basic_functions import ppaths, my_pool, Path, printdf, name_to_box
import rasterio as rio
import numpy as np
import shapely

def prod(lst):
    prod = 1
    for item in lst:
        prod *= item
    return prod


def check_sentinel_data(file_paths: list[Path]):
    for file_path in file_paths:
        with rio.open(file_path) as rio_f:
            if rio_f.shape != (832, 832):
                print(file_path)
        #     data = rio_f.read()
        #     num_zeros = len(np.where(data == 0)[0])
        #     total = prod(data.shape)
        # if num_zeros/total > .01:
        #     print(file_path.name)


def make_check_df(file_names: list[str], path_name_list: list[tuple] = None):
    if path_name_list is None:
        path_name_list = [
            (ppaths.waterway/'sentinel', 'sentinel'), (ppaths.waterway/'elevation_cut', 'elevation'),
            (ppaths.waterway/'waterways_burned', 'waterways'), (ppaths.waterway/'trails_burned', 'trails')
        ]
    data = {'file_name': [], 'geometry': []}
    for file_name in file_names:
        data['geometry'].append(name_to_box(file_name))
        data['file_name'].append(file_name)
        for path, name in path_name_list:
            data_list = data.setdefault(name, [])
            data_list.append((path/file_name).exists())
    df = gpd.GeoDataFrame(data, crs=4326)
    return df


def add_missing_path_files(file_name_list):
    waterways_dir = ppaths.waterway/'waterways_burned'
    trails_dir = ppaths.waterway/'trails_burned'
    for file_name in file_name_list:
        # print(file_name, (waterways_dir/file_name).exists(), (trails_dir/file_name).exists())
        with rio.open(waterways_dir/file_name) as rio_f:
            profile = rio_f.profile
            rows, cols = rio_f.shape
        array = np.zeros((1, rows, cols), dtype=np.uint8)
        with rio.open(trails_dir/file_name, 'w', **profile) as dst:
            dst.write(array)




if __name__ == '__main__':
    path_name_list = [
        (ppaths.waterway/'sentinel', 'sentinel'), (ppaths.waterway/'elevation_cut', 'elevation'),
        (ppaths.waterway/'waterways_burned', 'waterways'),
    ]

    files = list((ppaths.waterway/'sentinel').iterdir())
    # files = list((ppaths.country_data/'united_states/grid').iterdir())
    df = make_check_df(file_names=[file.name for file in files], path_name_list=path_name_list)
    point = shapely.Point((-91.973, 38.691))
    printdf(df[df.intersects(point)], 100)
