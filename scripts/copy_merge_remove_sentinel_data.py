import geopandas as gpd
import pandas as pd
import shapely
import rasterio as rio
import rioxarray as rxr
from water.basic_functions import ppaths, printdf, tt, time_elapsed, delete_directory_contents
import os
from concurrent.futures import ThreadPoolExecutor
# s = tt()
# country_data = ppaths.waterway/'country_data'
# world_bounds = gpd.read_file(ppaths.country_data/'world-administrative-boundaries.zip')
# world_geometry = shapely.unary_union(world_bounds.geometry.to_numpy())
# time_elapsed(s, 2)
# s = tt()
# world_tiles = gpd.read_parquet(country_data/'sentinel2_tile_shapes/tiles.parquet')
# print(len(world_tiles))
# world_tiles['geometry'] = world_tiles.buffer(-2e-7)
# time_elapsed(s, 2)
# s = tt()
# world_tile_tree = shapely.STRtree(world_tiles.geometry.to_numpy())
# world_tiles = world_tiles.loc[world_tile_tree.query(world_geometry, predicate='intersects')].reset_index(drop=True)
# time_elapsed(s, 2)
# s = tt()
# world_tiles['save_subdir'] = world_tiles.geometry.apply(lambda x: f'bbox_{x.bounds[0]}_{x.bounds[1]}_{x.bounds[2]}_{x.bounds[3]}')
# time_elapsed(s, 2)
# print(len(world_tiles))


def check_all_tile_locations(file_name):
    tile_types = ['africa', 'america', 'asia', 'europe', 'oceania', 'russia', 'usa']
    waterway_storage = ppaths.waterway/'waterway_storage'
    for tile_type in tile_types:
        tile_dir = waterway_storage / f'sentinel_tiles_{tile_type}'
        storage_path = tile_dir / file_name
        if storage_path.exists():
            delete_directory_contents(storage_path)
            storage_path.rmdir()

def delete_merged_tile(file):
    file_name = file.name.replace('.tif', '')
    check_all_tile_locations(file_name)


def delete_merged_tiles_from_storage(directory_path, max_workers=4):
    thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    file_list = directory_path.glob('*.tif')
    for file in file_list:
        thread_pool.submit(
            delete_merged_tile, **{'file': file}
        )


def copy_merged_file(file):
    waterway_storage_4326 = ppaths.waterway/'waterway_storage/sentinel_4326'
    os.system(f'cp {file} {waterway_storage_4326}')
    os.remove(file)


def copy_merged_files_to_storage(directory_path, max_workers=4):
    thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    file_list = directory_path.glob('*.tif')
    for file in file_list:
        thread_pool.submit(
            copy_merged_file, **{'file': file}
        )

if __name__ == '__main__':
    sentinel_4326_path = ppaths.country_data/'sentinel_4326'
    # delete_merged_tiles_from_storage(sentinel_4326_path, max_workers=10)
    copy_merged_files_to_storage(sentinel_4326_path)

