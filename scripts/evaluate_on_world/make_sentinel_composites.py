import os
from pathlib import Path
from water.basic_functions import ppaths, delete_directory_contents, tt, time_elapsed
import geopandas as gpd
import shapely
from multiprocessing import Process
import numpy as np
import rasterio as rio
from rasterio import transform
from rasterio.warp import Resampling

from water.make_country_waterways.cut_data import cut_data_to_image_and_save
from functools import cached_property

def copy_sentinel_from_storage(
        polygon: shapely.Polygon, sentinel_gdf: gpd.GeoDataFrame, sentinel_tree: shapely.STRtree
) -> None:
    file_indices = sentinel_tree.query(polygon, 'intersects')
    files_to_copy = sentinel_gdf.loc[file_indices]
    print(f"copying {len(files_to_copy)} files from storage", len(file_indices))
    temp_dir = ppaths.country_data/'sentinel_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = ppaths.storage_data/'sentinel_4326'
    sentinel_dir = ppaths.country_data/'sentinel_4326'
    for file in files_to_copy.file_name:
        if not (sentinel_dir/file).exists():
            os.system(f'cp {storage_dir/file} {temp_dir/file}')


def remove_current_sentinel_data(
        polygon: shapely.Polygon, sentinel_gdf: gpd.GeoDataFrame, sentinel_tree: shapely.STRtree
) -> None:
    files_to_keep = set(sentinel_gdf.loc[sentinel_tree.query(polygon, 'intersects')].file_name)
    sentinel_dir = ppaths.country_data/'sentinel_4326'
    for file in sentinel_dir.iterdir():
        if file.name not in files_to_keep:
            os.remove(file)


def move_temp_sentinel_data_to_sentinel():
    sentinel_dir = ppaths.country_data/'sentinel_4326'
    temp_dir = ppaths.country_data/'sentinel_temp'
    for file in temp_dir.iterdir():
        file_name = file.name
        new_path = sentinel_dir/file_name
        file.rename(new_path)


def make_temp_tile(tile_geometry: shapely.Polygon, save_path: Path, grid_shape: tuple[int, int]):
    if not save_path.exists():
        grid_width, grid_height = grid_shape
        min_lon, min_lat, max_lon, max_lat = tile_geometry.bounds
        trans = transform.from_bounds(min_lon, min_lat, max_lon, max_lat, grid_width, grid_width)
        data = np.zeros((1, grid_width, grid_width), dtype=np.uint8)
        meta = {
            'driver': 'GTiff', 'dtype': 'uint8', 'nodata': 0,
            'width': grid_width, 'height': grid_width, 'count': 1,
            'crs': rio.CRS.from_epsg(4326), 'transform': trans
        }
        with rio.open(save_path, 'w', **meta) as dst_f:
            dst_f.write(data)


def merge_sentinel_data_and_remove_temp(save_dir: Path, temp_file_path: Path):
    cut_data_to_image_and_save(
        save_dir=save_dir, image_paths=[temp_file_path], data_dir=ppaths.country_data/'sentinel_4326',
        resampling=Resampling.bilinear
    )
    os.remove(temp_file_path)


class SentinelCompositeMaker:
    def __init__(
            self, zoom_level: int, temp_grid_dir: Path = ppaths.country_data/'temp_grids',
            save_directory: Path = ppaths.country_data/f'sentinel_merged'
    ):
        self.zoom_level = zoom_level
        self.grid_shape = (256*2**(13-zoom_level), 256*2**(13-zoom_level))
        self.sentinel_storage_info = gpd.read_parquet(ppaths.storage_data/'sentinel_4326/sentinel_4326.parquet')
        self.sentinel_storage_tree = shapely.STRtree(self.sentinel_storage_info.geometry.to_numpy())
        self.temp_grid_dir = temp_grid_dir
        self.save_directory = save_directory
        self.current_index = 0
        self.input_list = list(zip(self.polygon_info.x_index, self.polygon_info.y_index, self.polygon_info.geometry))
        self.times = []

    @cached_property
    def polygon_info(self):
        tile_gdf = gpd.read_parquet(ppaths.country_lookup_data/f'zoom_level_{self.zoom_level}_xyz_info.parquet')
        world_boundaries = gpd.read_parquet(ppaths.country_lookup_data/'world_boundaries.parquet')
        world_boundaries = world_boundaries[world_boundaries.continent == 'Africa']
        world_polygon = shapely.unary_union(world_boundaries.geometry.to_numpy()).buffer(.05)
        # tile_gdf = tile_gdf[(73 <= tile_gdf.x_index) & (tile_gdf.x_index <= 73)
        #                     & (64 <= tile_gdf.y_index) & (tile_gdf.y_index <= 70)]
        polygon_info = tile_gdf[tile_gdf.intersects(world_polygon)].reset_index(drop=True)
        print('number of tiles', len(polygon_info))
        polygon_info = polygon_info.sort_values(by=['x_index', 'y_index']).reset_index(drop=True)
        polygon_info['file_exists'] = polygon_info[['x_index', 'y_index']].apply(
            lambda row: (self.save_directory/f'{row.x_index}_{row.y_index}.tif').exists(), axis=1
        )
        return polygon_info[~polygon_info.file_exists].reset_index(drop=True)

    def make_copy_inputs(self, index):
        _, _, geometry = self.input_list[index]
        copy_inputs = dict(
            polygon=geometry, sentinel_gdf=self.sentinel_storage_info, sentinel_tree=self.sentinel_storage_tree
        )
        return copy_inputs

    def make_temp_inputs(self, index):
        x_index, y_index, geometry = self.input_list[index]
        save_path = self.temp_grid_dir / f'{x_index}_{y_index}.tif'
        temp_inputs = dict(tile_geometry=geometry, save_path=save_path, grid_shape=self.grid_shape)
        return temp_inputs

    def make_merge_inputs(self, index):
        x_index, y_index, geometry = self.input_list[index]
        temp_file_path = self.temp_grid_dir / f'{x_index}_{y_index}.tif'
        merge_inputs = dict(save_dir=self.save_directory, temp_file_path=temp_file_path)
        return merge_inputs

    def make_next_inputs(self):
        next_index = self.current_index + 1 if self.current_index + 1 < len(self.input_list) else self.current_index
        return (self.make_merge_inputs(self.current_index), self.make_copy_inputs(next_index),
                self.make_temp_inputs(next_index))

    def run_for_current_inputs(self):
        merge_inputs, copy_inputs, make_temp_inputs = self.make_next_inputs()
        merge_process = Process(target=merge_sentinel_data_and_remove_temp, kwargs=merge_inputs)
        copy_process = Process(target=copy_sentinel_from_storage, kwargs=copy_inputs)
        make_temp_process = Process(target=make_temp_tile, kwargs=make_temp_inputs)
        merge_process.start()
        copy_process.start()
        make_temp_process.start()
        merge_process.join()
        copy_process.join()
        make_temp_process.join()
        merge_process.close()
        copy_process.close()
        make_temp_process.close()
        remove_current_sentinel_data(**copy_inputs)
        move_temp_sentinel_data_to_sentinel()
        self.current_index += 1

    def main(self):
        copy_inputs = self.make_copy_inputs(0)
        temp_inputs = self.make_temp_inputs(0)
        print('Making init inputs')
        s = tt()
        make_temp_tile(**temp_inputs)
        copy_sentinel_from_storage(**copy_inputs)
        remove_current_sentinel_data(**copy_inputs)
        move_temp_sentinel_data_to_sentinel()
        time_elapsed(s, 2)
        while self.current_index <= len(self.input_list):
            s_i = tt()
            self.run_for_current_inputs()
            time_elapsed(s, 2)
            self.times.append(tt() - s_i)
            time_elapsed(s, 2)
            mean_time = int(np.mean(self.times))
            print(f'  Mean time per iteration : {mean_time // 60}m, {mean_time % 60}s')
            print(f'  Completed {self.current_index}/{len(self.input_list)}')


if __name__ == '__main__':
    sentinel_composite_maker = SentinelCompositeMaker(zoom_level=7)
    sentinel_composite_maker.main()

