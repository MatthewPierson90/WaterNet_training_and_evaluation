import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely

from water.basic_functions import printdf, tt, time_elapsed, SharedMemoryPool
from water.paths import ppaths
from pprint import pprint


def open_geopandas_state_data(state:str, bbox=None):
    state = state.capitalize()
    state_path = ppaths.data/f'waterway_data/waterways_state/NHD_H_{state}_State_GPKG.zip!NHD_H_{state}_State_GPKG.gpkg'
    if bbox is not None:
        gdf = gpd.read_file(state_path, bbox=bbox)
    else:
        gdf = gpd.read_file(state_path)
    return gdf


def open_hu4_data(index:int):
    waterway_path = ppaths.data/f'waterway_data/waterways_hu4/NHD_H_{index:04d}_HU4_GPKG.zip!NHD_H_{index:04d}_HU4_GPKG.gpkg'
    gdf1 = gpd.read_file(waterway_path, layer='NHDFlowline')
    # printdf(gdf1)
    gdf1 = gdf1[['permanent_identifier','fdate', 'resolution','resolution_description',
                 'visibilityfilter', 'visibilityfilter_description',
                 'ftype','fcode','fcode_description', 'geometry']]
    gdf1['layer'] = 'NHDFlowline'

    gdf2 = gpd.read_file(waterway_path, layer='NHDWaterbody')
    # printdf(gdf2)

    gdf2 = gdf2[['permanent_identifier','fdate', 'resolution','resolution_description',
                 'visibilityfilter', 'visibilityfilter_description',
                 'ftype','fcode','fcode_description', 'geometry']]
    gdf2['layer'] = 'NHDWaterbody'

    gdf3 = gpd.read_file(waterway_path, layer='NHDArea')
    # printdf(gdf3)
    gdf3 = gdf3[['permanent_identifier','fdate', 'resolution','resolution_description',
                 'visibilityfilter', 'visibilityfilter_description',
                 'ftype','fcode','fcode_description', 'geometry']]
    gdf3['layer'] = 'NHDArea'

    gdf4 = gpd.read_file(waterway_path, layer='NHDLine')
    # printdf(gdf4)

    gdf4 = gdf4[['permanent_identifier','fdate', 'resolution','resolution_description', 'visibilityfilter',
                 'visibilityfilter_description', 'ftype','fcode','fcode_description', 'geometry']]
    gdf4['layer'] = 'NHDLine'

    gdf = pd.concat([gdf1, gdf2, gdf3, gdf4], ignore_index=True)
    return gdf


def hu4_to_parquet(index: int):
    gdf = open_hu4_data(index)
    gdf.to_parquet(ppaths.training_data/f'hu4_parquet/hu4_{index:04d}.parquet')


def multi_hu4_to_parquet(num_proc=12):
    inds = []
    for i in range(101, 2206):
        path = ppaths.training_data/f'waterways_hu4/NHD_H_{i:04d}_HU4_GPKG.zip'
        if path.exists():
            inds.append(i)
    if not (ppaths.training_data/'hu4_parquet').exists():
        (ppaths.training_data/'hu4_parquet').mkdir()
    SharedMemoryPool(func=hu4_to_parquet, input_list=inds, num_proc=num_proc).run()


def open_hu4_plus_data(index:int):
    waterway_path = ppaths.data/f'waterway_data/waterways_p_hu4/NHDPLUS_H_{index:04d}_HU4_GDB.zip!NHDPLUS_H_{index:04d}_HU4_GDB.gdb'
    gdf1 = gpd.read_file(waterway_path, layer='NHDFlowline')
    # printdf(gdf1)
    gdf1 = gdf1[['permanent_identifier','fdate', 'resolution','resolution_description',
                 'visibilityfilter', 'visibilityfilter_description',
                 'ftype','fcode','fcode_description', 'geometry']]
    gdf1['layer'] = 'NHDFlowline'

    gdf2 = gpd.read_file(waterway_path, layer='NHDWaterbody')
    # printdf(gdf2)

    gdf2 = gdf2[['permanent_identifier','fdate', 'resolution','resolution_description',
                 'visibilityfilter', 'visibilityfilter_description',
                 'ftype','fcode','fcode_description', 'geometry']]
    gdf2['layer'] = 'NHDWaterbody'

    gdf3 = gpd.read_file(waterway_path, layer='NHDArea')
    # printdf(gdf3)
    gdf3 = gdf3[['permanent_identifier','fdate', 'resolution','resolution_description',
                 'visibilityfilter', 'visibilityfilter_description',
                 'ftype','fcode','fcode_description', 'geometry']]
    gdf3['layer'] = 'NHDArea'

    gdf4 = gpd.read_file(waterway_path, layer='NHDLine')
    # printdf(gdf4)

    gdf4 = gdf4[['permanent_identifier','fdate', 'resolution','resolution_description', 'visibilityfilter',
                 'visibilityfilter_description', 'ftype','fcode','fcode_description', 'geometry']]
    gdf4['layer'] = 'NHDLine'

    gdf = pd.concat([gdf1, gdf2, gdf3, gdf4], ignore_index=True)
    return gdf


def open_hu4_parquet_data(index):
    return gpd.read_parquet(ppaths.training_data/f'hu4_parquet/hu4_{index:04d}.parquet')

def open_hu4_hull_data(index):
    return gpd.read_parquet(ppaths.training_data/f'hu4_hull/hu4_{index:04d}.parquet')

def make_hu4_linestring_hull(index):
    print(f'working on {index}')
    file_path = ppaths.training_data/f'hu4_hull/hu4_{index:04d}.parquet'
    og_path = ppaths.training_data/f'hu4_parquet/hu4_{index:04d}.parquet'
    if not file_path.exists() and og_path.exists():
        gdf = open_hu4_parquet_data(index)
        linestrings = gdf[gdf.type.str.contains('LineString')].geometry.to_list()
        geometry = shapely.concave_hull(shapely.union_all(linestrings), .01)
        hull_gdf = gpd.GeoDataFrame({'geometry': [geometry]}, crs=gdf.crs)
        hull_gdf.to_parquet(file_path)
        print(f'Completed hu4 {index:04d}')

def make_all_hu4_hulls(num_proc=2):
    if not (ppaths.training_data/'hu4_hull').exists():
        (ppaths.training_data/'hu4_hull').mkdir()
    s = tt()
    base_dir = ppaths.training_data/f'hu4_parquet'
    hull_dir = ppaths.training_data/'hu4_hull'
    inputs = [
        index for index in range(706, 1900) if (base_dir/f'hu4_{index:04d}.parquet').exists()
    ]
    if num_proc > 1:
        SharedMemoryPool(num_proc=num_proc, func=make_hu4_linestring_hull, input_list=inputs).run()
    else:
        map(make_hu4_linestring_hull, inputs)

if __name__ == '__main__':
    from pprint import pprint
    import fiona
    file_path = ppaths.training_data/'WBD_National_GDB/WBD_National_GDB.gdb'
    layers = fiona.listlayers(file_path)
    gdf = gpd.read_file(file_path, layer='WBDHU4')