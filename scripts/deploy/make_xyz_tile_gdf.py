import os

import numpy as np
import shapely
import geopandas as gpd
from water.paths import ppaths


def tile_to_box(x_index, y_index, zoom):
    zoom_scale = 2 ** zoom
    x_min = x_index / zoom_scale * 360.0 - 180.0
    x_max = (x_index + 1) / zoom_scale * 360.0 - 180.0
    y_max = np.arctan(np.sinh(np.pi*(1 - 2 * y_index / zoom_scale)))*180/np.pi
    y_min = np.arctan(np.sinh(np.pi*(1 - 2 * (y_index + 1) / zoom_scale)))*180/np.pi
    return x_min, y_min, x_max, y_max


def make_tile_gdf(zoom_level):
    zoom_scale = 2 ** zoom_level
    x_indices = np.concatenate([np.array([i]*zoom_scale, dtype=np.uint16) for i in range(zoom_scale)])
    y_indices = np.concatenate([np.array([i for i in range(zoom_scale)], dtype=np.uint16)]*zoom_scale)
    x_min = x_indices.astype(np.float32) / zoom_scale * 360.0 - 180
    x_max = (x_indices.astype(np.float32) + 1) / zoom_scale * 360.0 - 180
    y_min = np.arctan(np.sinh(np.pi*(1 - 2 * y_indices.astype(np.float32) / zoom_scale)))*180/np.pi
    y_max = np.arctan(np.sinh(np.pi*(1 - 2 * (y_indices.astype(np.float32) + 1) / zoom_scale)))*180/np.pi
    x_min = np.round(x_min, 9)
    x_max = np.round(x_max, 9)
    y_min = np.round(y_min, 9)
    y_max = np.round(y_max, 9)
    bboxes = shapely.box(x_min, y_min, x_max, y_max)
    gdf = gpd.GeoDataFrame(
        data={'x_index': x_indices, 'y_index': y_indices, 'geometry': bboxes},
        crs=4326
    )
    return gdf


def open_tile_gdf(zoom_level):
    if not (ppaths.world_info/f'zoom_level_{zoom_level}_xyz_info.parquet').exists():
        gdf = make_tile_gdf(zoom_level)
        gdf.to_parquet(ppaths.world_info/f'zoom_level_{zoom_level}_xyz_info.parquet')
    else:
        gdf = gpd.read_parquet(ppaths.world_info/f'zoom_level_{zoom_level}_xyz_info.parquet')
    return gdf


if __name__ == '__main__':
    open_tile_gdf(6)
