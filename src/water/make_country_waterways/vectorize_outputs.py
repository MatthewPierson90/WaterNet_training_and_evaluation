import matplotlib.pyplot as plt
import pandas as pd

from water.raster_to_vector.vectorize import Vectorizer
from water.basic_functions import ppaths, my_pool, delete_directory_contents
import geopandas as gpd
import shapely


def vectorize_file(file, elevation_file, save_dir, small_island_count, min_water_val, save_name=None, tif_index=0):
    vector = Vectorizer(
        file, elevation_file=elevation_file, small_land_count=small_island_count,
        min_water_val=min_water_val, tif_index=tif_index
    )
    lines = gpd.GeoDataFrame({'geometry': vector.line_strings}, crs=4326)
    if save_name is None:
        save_name = file.name.split('.tif')[0]+'.parquet'
    lines.to_parquet(save_dir/save_name)


def vectorize_and_save(
        country, country_dir, num_proc=14, output_dir_name='output_data',
        small_land_count=40, min_water_val=.5, save_name='waterways_model', tif_index=0
):
    save_name = f'{country}_{save_name}.parquet'
    output_dir = country_dir/output_dir_name
    save_dir = country_dir/'vector_data'
    elevation_file = country_dir/f'elevation_merged/{country}_merged_data.tif'
    if save_dir.exists():
        delete_directory_contents(save_dir)
    else:
        save_dir.mkdir()
    files_to_vectorize = output_dir.glob('*.tif')
    inputs = [
        {
            'file': file, 'elevation_file': elevation_file, 'save_dir': save_dir,
            'small_island_count': small_land_count, 'min_water_val': min_water_val, 'tif_index': tif_index
        } for file in files_to_vectorize
    ]
    if num_proc > 1:
        my_pool(num_proc=num_proc, func=vectorize_file, input_list=inputs, use_kwargs=True)
    else:
        vectorize_file(**inputs[0])
    vc = save_dir.glob('*')
    dfs = [gpd.read_parquet(file) for file in vc]
    if len(dfs) > 1:
        df = pd.concat(dfs, ignore_index=True)
        df_geoms = shapely.line_merge(shapely.MultiLineString(df.geometry.to_list()))
        df_geoms = shapely.node(df_geoms)
        df = gpd.GeoDataFrame({'geometry': df_geoms.geoms}, crs=4326)
    else:
        df = dfs[0]
    # adm_0 = gpd.read_file(country_dir/f'{country}_admin_boundaries')
    save_path = country_dir/save_name
    if save_path.exists():
        save_name_split = save_name.split('.parquet')[0]
        old_count = get_model_count(country, save_name_split) + 1
        old_move = country_dir / f'{save_name_split}_{old_count}.parquet'
        save_path.rename(old_move)
    df.to_parquet(save_path)

def get_model_count(country, save_name_split):
    country_dir = ppaths.waterway/f'country_data/{country}'
    old_count = len(list(country_dir.glob(f'{save_name_split}*parquet'))) - 1
    return old_count

def plot_country_data(country, ax=None):
    country_data = ppaths.waterway / 'country_data'
    country_path = country_data / country
    df = gpd.read_parquet(ppaths.waterway / f'country_data/{country}/{country}_waterways_model.parquet')
    df = df.reset_index(drop=True)
    adm_0 = get_country_polygon(country)

    bridges = gpd.read_parquet(country_data / f'bridge_locations.parquet')
    bridges = bridges[bridges.intersects(adm_0)]
    strt = shapely.STRtree(df.geometry)
    df = df.loc[strt.query(adm_0, 'intersects')]
    ax = df.plot(ax=ax)

    # if (country_path/f'{country}_natural_osm.parquet').exists():
    #     natural = gpd.read_parquet(country_path / f'{country}_natural_osm.parquet')
    #     natural = natural[(~natural.water.isna()) | (~natural.wetland.isna())]
    #     natural.plot(ax=ax, color='aqua')
    # if (country_path / f'geoglows_ww.parquet').exists():
    #     osm_df = gpd.read_parquet(country_path / f'geoglows_ww.parquet')
    #     osm_df.plot(ax=ax, color='lime')
    if (country_path / f'{country}_waterways_osm.parquet').exists():
        osm_df = gpd.read_parquet(country_path / f'{country}_waterways_osm.parquet')
        osm_df.plot(ax=ax, color='aqua')
    #     ww = shapely.MultiLineString(osm_df.geometry.to_list())

    if (country_path / f'{country}_footpaths_osm_recut.parquet').exists():
        osm_df = gpd.read_parquet(country_path / f'{country}_footpaths_osm_recut.parquet')
        footpaths = shapely.MultiLineString(osm_df.geometry.to_list())
        ww = shapely.MultiLineString(df.geometry.to_list())
        # intersections = footpaths.intersection(ww)
        # gpd.GeoSeries([intersections], crs=4326).plot(ax=ax, color='orange')
        osm_df.plot(ax=ax, color='orange')
    # if len(bridges) > 0:
    #     bridges.plot(ax=ax, color='red')
    ax.set_facecolor('black')


def save_shapefiles(country):
    country_path = ppaths.country_data/country
    shape_path = country_path/'shapefiles'
    if not shape_path.exists():
        shape_path.mkdir()
    model_df = gpd.read_parquet(country_path/f'{country}_waterways_model.parquet')
    # osm_df = gpd.read_parquet(country_path/f'{country}_waterways_osm.parquet')
    # osm_foot = gpd.read_parquet(country_path/f'{country}_footpaths_osm_recut.parquet')
    # natural = gpd.read_parquet(country_path / f'{country}_natural_osm.parquet')
    # natural = natural[(~natural.water.isna()) | (~natural.wetland.isna())]
    adm_0 = get_country_polygon(country)
    # bridges = gpd.read_parquet(ppaths.country_data / f'bridge_locations.parquet')
    # bridges = bridges[bridges.intersects(adm_0)]
    model_df = model_df[model_df.intersects(adm_0)].reset_index(drop=True)
    model_df.to_file(shape_path/f'{country}_waterways_model')
    # osm_df[['geometry']].to_file(shape_path/f'{country}_waterways_osm')
    # osm_foot[['geometry']].to_file(shape_path/f'{country}_footpaths_osm')
    # natural[['geometry']].to_file(shape_path/f'{country}_natural_osm')
    # bridges[['geometry']].to_file(shape_path/f'{country}_bridges')


def compare_model_outputs(country: str, index1: int = None, index2: int = None, plot_2_first: bool = True):
    country_dir = ppaths.country_data/country
    if index1 is None:
        new = gpd.read_parquet(country_dir/f'{country}_waterways_model.parquet')
    else:
        new = gpd.read_parquet(country_dir/f'{country}_waterways_model_{index1}.parquet')
    if index2 is None:
        index2 = get_model_count(country)
    old = gpd.read_parquet(country_dir/f'{country}_waterways_model_{index2}.parquet')
    dfs = [new, old]
    colors = ['aqua', 'orange']
    if plot_2_first:
        dfs.reverse()
        colors.reverse()
    fig, ax = plt.subplots()
    for df, color in zip(dfs, colors):
        df.plot(ax=ax, color=color, legend=True)
    ax.set_facecolor('black')

if __name__ == '__main__':
    from water.basic_functions import get_country_polygon
    import matplotlib
    import rasterio as rio
    matplotlib.use('TkAgg')
    # ctry = 'rwanda'
    # ctry = 'zambia'
    # countries = [
    #     'germany', 'france', 'belgium', 'netherlands', 'switzerland', 'austria', 'italy', 'spain',
    #     'burundi', 'liberia', 'panama', 'nicaragua', 'bolivia', 'zambia', 'ethiopia',  'somalia',
    #     'south_sudan', 'rwanda', 'uganda', 'ivory_coast', 'kenya', 'tanzania'
    # ]
    countries3 = [
        'rwanda', 'uganda', 'ivory_coast', 'germany', 'france', 'belgium',
        'netherlands', 'switzerland', 'austria', 'italy', 'spain',
        'burundi', 'liberia', 'panama', 'nicaragua', 'bolivia', 'zambia',
        'ethiopia',  'somalia', 'south_sudan', 'kenya', 'tanzania'
    ]
    for ctry in countries3:
        save_name = 'waterways_model'
        country_dir = ppaths.country_data/f'{ctry}'
        save_name = f'{ctry}_{save_name}.parquet'
        save_path = country_dir/save_name

        vectorize_file()
    # countries = [
    #     'bolivia', 'zambia', 'ethiopia',  'somalia', 'south_sudan', 'rwanda', 'uganda', 'ivory_coast'
    # ]
    # countries = ['kenya', 'tanzania']
    # countries = ['rwanda']

    # for ctry in countries:
    #     vectorize_and_save(
    #         country=ctry, num_proc=1, output_dir_name='output_data_merged', min_water_val=.4, small_land_count=25
    #     )

    # print(get_model_count(ctry))
    # compare_model_outputs(ctry, plot_2_first=False, index1=None, index2=None)
    # plot_country_data(country=ctry)


