import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely

from water.basic_functions import ppaths, printdf, tt, time_elapsed, my_pool
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
    gdf.to_parquet(ppaths.waterway/f'hu4_parquet/hu4_{index:04d}.parquet')


def multi_hu4_to_parquet(num_proc=12):
    inds = []
    for i in range(101, 2206):
        path = ppaths.waterway/f'waterways_hu4/NHD_H_{i:04d}_HU4_GPKG.zip'
        if path.exists():
            inds.append(i)
    if not (ppaths.waterway/'hu4_parquet').exists():
        (ppaths.waterway/'hu4_parquet').mkdir()
    my_pool(func=hu4_to_parquet, input_list=inds, num_proc=num_proc)


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
    return gpd.read_parquet(ppaths.waterway/f'hu4_parquet/hu4_{index:04d}.parquet')

def open_hu4_hull_data(index):
    return gpd.read_parquet(ppaths.waterway/f'hu4_hull/hu4_{index:04d}.parquet')

def make_hu4_linestring_hull(index):
    print(f'working on {index}')
    file_path = ppaths.waterway/f'hu4_hull/hu4_{index:04d}.parquet'
    og_path = ppaths.waterway/f'hu4_parquet/hu4_{index:04d}.parquet'
    if not file_path.exists() and og_path.exists():
        gdf = open_hu4_parquet_data(index)
        linestrings = gdf[gdf.type.str.contains('LineString')].geometry.to_list()
        geometry = shapely.concave_hull(shapely.union_all(linestrings), .01)
        hull_gdf = gpd.GeoDataFrame({'geometry': [geometry]}, crs=gdf.crs)
        hull_gdf.to_parquet(file_path)
        print(f'Completed hu4 {index:04d}')

def make_all_hu4_hulls(num_proc=2):
    if not (ppaths.waterway/'hu4_hull').exists():
        (ppaths.waterway/'hu4_hull').mkdir()
    s = tt()
    base_dir = ppaths.waterway/f'hu4_parquet'
    hull_dir = ppaths.waterway/'hu4_hull'
    inputs = [
        index for index in range(706, 1900) if (base_dir/f'hu4_{index:04d}.parquet').exists()
    ]
    if num_proc > 1:
        my_pool(num_proc=num_proc, func=make_hu4_linestring_hull, input_list=inputs)
    else:
        map(make_hu4_linestring_hull, inputs)

if __name__ == '__main__':
    import pyproj
    from pprint import pprint
    from shapely.ops import transform as sh_transform
    # make_all_hu4_hulls(1)
    fcodes = {}
    # gdf = open_hu4_parquet_data(index=101)
    # d = gdf.groupby('fcode_description').layer.count().to_dict()
    # pprint(d)
    # hu4_data = {'hu4_index': [], 'geometry': []}
    # for index in range(101, 102):
    for index in range(101, 1900):
        if (ppaths.waterway/f'hu4_parquet/hu4_{index:04d}.parquet').exists():
            # gdf = open_hu4_hull_data(index)
            # hu4_data['hu4_index'].append(index)
            # hu4_data['geometry'].append(gdf.geometry[0])
    # hu4_df = gpd.GeoDataFrame(hu4_data, crs=4326)
            print(f'getting index {index}')
            gdf = open_hu4_parquet_data(index)
            # printdf(gdf)
            # dct = gdf.groupby('fcode_description').layer.count().to_dict()
            # for item, value in dct.items():
            #     if item in fcodes:
            #         fcodes[item] += value
            #     else:
            #         fcodes[item] = value
            dct = gdf.groupby('visibilityfilter').layer.count().to_dict()
            for item, value in dct.items():
                if item in fcodes:
                    fcodes[item] += value
                else:
                    fcodes[item] = value
            # fcodes.update(gdf.fcode_description.unique())
    # printdf(gdf, 200)
    pprint(fcodes)
    # printdf(gdf.groupby('fcode_description')[['layer', 'geometry']].count(), 10000)
    # fig, ax = plt.subplots()
    
    # inter = gdf[gdf.fcode_description.str.contains('Intermittent')]
    # emph = gdf[gdf.fcode_description.str.contains('Ephemeral')]
    # perennial = gdf[(gdf.fcode_description.str.contains('Perennial')) | (gdf.fcode_description == 'Stream/River')]
    # other = gdf[~(gdf.fcode_description.str.contains('Intermittent') | gdf.fcode_description.str.contains('Ephemeral') | gdf.fcode_description.str.contains('Perennial') | (gdf.fcode_description == 'Stream/River'))]
    # perennial.plot(ax=ax)
    # other.plot(ax=ax, color='aqua')
    # if len(inter) > 0:
    #     inter.plot(ax=ax, color='red')
    # if len(emph)>0:
    #     emph.plot(ax=ax, color='green')
    # ax.set_facecolor('black')
    # print(gdf.fcode_description.unique())
    # printdf(gdf[[col for col in gdf.columns if col != 'geometry']])
    # printdf(gdf.groupby('fcode_description').resolution.count(),
    #         len(gdf.fcode_description.unique())+5)
    # print(gdf.type.unique())
    # print('linestrings')
    # s =tt()
    # linestrings = shapely.concave_hull(shapely.union_all(gdf[gdf.type.isin(['MultiLineString', 'LineString'])].geometry.to_list()), .01)
    # time_elapsed(s, 2)
    # print('polygons')
    # s=tt()
    # polygons = shapely.concave_hull(shapely.union_all(gdf[gdf.type.isin(['Polygon', 'MultiPolygon'])].geometry.to_list()), .5)
    # time_elapsed(s, 2)
    # print('union')
    # s = tt()
    # union = shapely.union(linestrings, polygons)
    # time_elapsed(s, 2)
    # print('hull')
    # s = tt()
    # # waterways_hull = shapely.concave_hull(union)
    # time_elapsed(s, 2)
    # ax = gpd.GeoSeries([linestrings], crs=4269).plot()
    # gdf.plot(ax=ax, color='red')
    # file_path = ppaths.waterway/'rivers_4269.parquet'
    # file_path = ppaths.waterway/'detailed_rivers.parquet'
    # gdf = gpd.read_parquet(file_path)
    # print(gdf)
    # # print(gdf.crs)
    # geom_0 = gdf.loc[1, 'geometry']
    # print(geom_0)
    # # from_crs = pyproj.crs.CRS.from_epsg(4326)
    # to_crs = pyproj.crs.CRS.from_epsg(4269)
    # transformer = pyproj.Transformer.from_crs(crs_from=gdf.crs, crs_to=to_crs, always_xy=True)
    # geom_t = sh_transform(transformer.transform, geom_0)
    # print(geom_t)

    # try:
    #     gdf
    # except:
    #     gdf = open_hu4_data(101)
    #     print(gdf.dtypes)
    # print(gdf[gdf.layer == 'NHDFlowline'].fdate.min(), gdf[gdf.layer == 'NHDFlowline'].fdate.max())
    # multi_hu4_to_parquet(12)
    import fiona
    # index = 101
    # file = ppaths.data/f'waterway_data/waterways_hu4/NHD_H_{index:04d}_HU4_GPKG/NHD_H_{index:04d}_HU4_GPKG.gpkg'
    # layers = fiona.listlayers(file)
    # pprint(layers)
    # try:
    #     gdf
    # except:
        # bb = (-105.694362, 39.912886, -105.052774, 40.262785)
        # gdf = open_geopandas_state_data('colorado', bbox=bb)
    # gdf = open_hu4_plus_data(1019)
    # s = tt()
    # gdf = open_hu4_data(101)
    # time_elapsed(s)
    # gdf.plot()

    #
    # s = tt()
    # gdf = gpd.read_file(ppaths.waterway/f'NHD_H_0101_HU4_Shape.zip!Shape/NHDFlowline.shp')
    # time_elapsed(s)
    # s=tt()
    # gdf.to_parquet(ppaths.waterway/f'NHD_H_0101_HU4_NHDFlowline.parquet')
    # time_elapsed(s)
    #
    # s=tt()
    # gdf = gpd.read_parquet(ppaths.waterway/f'NHD_H_0101_HU4_NHDFlowline.parquet')
    # time_elapsed(s)

    # print(gdf.columns)
    # printdf(gdf[[col for col in gdf.columns if col!='geometry']],10)
    # printdf(gdf,10)

    # gdf.plot()
    # gdf.type.unique()
    # print(gdf.total_bounds)