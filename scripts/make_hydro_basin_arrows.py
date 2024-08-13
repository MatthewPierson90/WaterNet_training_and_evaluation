import geopandas as gpd
from water.basic_functions import ppaths, printdf, SharedMemoryPool
import shapely
import random

hydro_abbrevs = {
    'af': 'africa',
    'ar': 'arctic',
    'as': 'asia',
    'au': 'australia',
    'eu': 'europe',
    'gr': 'greenland',
    'na': 'north_america',
    'sa': 'south_america',
    'si': 'siberia'
}

def open_hydrobasins_shapefile(area: str, level: int):
    hydrobasins_path = ppaths.training_data/'hydrobasins/'
    return gpd.read_file(hydrobasins_path/f'hybas_lake_{area}_lev01-12_v1c/hybas_lake_{area}_lev{level:02d}_v1c.shp')


def make_parquet_file(area: str, level: int):
    save_path = ppaths.training_data/f'hydrobasins_parquet/{hydro_abbrevs[area]}_level_{level}.parquet/'
    if not save_path.exists():
        hydro_df = open_hydrobasins_shapefile(area, level)
        hydro_df.to_parquet(save_path)


def open_hydrobasins_parquet(area: str, level: int):
    path = ppaths.training_data/f'hydrobasins_parquet/{area}_level_{level}.parquet/'
    return gpd.read_parquet(path)


def make_line(point, next_index, hydro_df):
    if next_index == 0:
        return point
    return shapely.LineString([point, hydro_df.loc[next_index, 'geometry']])


def make_hydro_arrow(area, level):
    hydro_df = open_hydrobasins_parquet(area, level)
    replace_indices = hydro_df[hydro_df.ENDO == 2].index
    hydro_df.loc[replace_indices, 'NEXT_DOWN'] = 0
    hydro_points = hydro_df.copy()
    hydro_points['geometry'] = shapely.point_on_surface(hydro_df.geometry.to_numpy())
    hydro_points = hydro_points.set_index('HYBAS_ID')
    hydro_lines = hydro_points.copy()
    hydro_lines['geometry'] = hydro_points[['geometry', 'NEXT_DOWN']].apply(
        lambda x: make_line(x.geometry, x.NEXT_DOWN, hydro_points), axis=1
    )
    hydro_lines_path = ppaths.training_data/f'hydrobasins_lines/{area}_level_{level}_lines.parquet/'
    hydro_lines.to_parquet(hydro_lines_path)


if __name__ == "__main__":
    levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    inputs = [
        dict(area=area, level=level) for area in hydro_abbrevs.values() for level in levels
    ]
    random.shuffle(inputs)
    SharedMemoryPool(num_proc=10, func=make_hydro_arrow, input_list=inputs, use_kwargs=True).run()
    # hydro_df = open_hydrobasins_shapefile('af', 7)
    # replace_indices = hydro_df[hydro_df.ENDO == 2].index
    # hydro_df.loc[replace_indices, 'NEXT_DOWN'] = 0
    # hydro_points = hydro_df.copy()
    # hydro_points['geometry'] = shapely.point_on_surface(hydro_df.geometry.to_numpy())
    # hydro_points = hydro_points.set_index('HYBAS_ID')
    # hydro_lines = hydro_points.copy()
    # hydro_lines['geometry'] = hydro_points[['geometry', 'NEXT_DOWN']].apply(lambda x: make_line(x.geometry, x.NEXT_DOWN, hydro_points), axis=1)
    # # ax = hydro_df.exterior.plot()
    # hydro_points.plot(ax=ax, color='red')
    # hydro_lines.plot(ax=ax, color='green')