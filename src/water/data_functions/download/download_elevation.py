import shapely

from water.basic_functions import (ppaths, get_country_bounding_box,
                                   tt, time_elapsed, bboxtype, my_pool)
import logging
import pystac_client
import requests
import geopandas as gpd
from pathlib import Path


def query_pc(bbox: bboxtype,
             collections: list,
             datetime: str = None,
             max_items: int = None
             ) -> list:
    """
    Query the microsoft planetary for data in the entered collection, bbox, and datetime
    Parameters
    ----------
    bbox: tuple
    collections: list, str
    datetime: str
    max_items: int

    Returns
    -------

    """
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    logging.info('Searching Catalog')
    search = catalog.search(collections=collections,
                            bbox=bbox,
                            datetime=datetime,
                            max_items=max_items, )
    items = list(search.get_items())
    # print(len(items))
    return items


def download_and_save_single_file(input_dict):
    item = input_dict['item']
    save_name = input_dict['save_name']
    save_dir = input_dict['save_dir']
    force_download = input_dict['force_download']
    url = item.assets['data'].href
    item_bbox = item.bbox
    save_path = save_dir/f'{save_name}_{item_bbox[0]}_{item_bbox[1]}_{item_bbox[2]}_{item_bbox[3]}.tif'
    retry_count = 0
    if not save_path.exists() or force_download:
        response = requests.get(url, timeout=20)
        with open(save_path, 'wb') as f:
            f.write(response.content)


def download_bbox_elevation_data(bbox: bboxtype,
                                 save_dir: Path = ppaths.country_data/'elevation',
                                 num_proc: int = 2,
                                 force_download: bool = False,
                                 print_progress: bool = True
                                 ) -> None:
    """
    Downloads elevation data for the entered bbox, and saves the data to
    model_data/country/country_elevation/country_elevation_parts
    """
    items = query_pc(bbox=bbox,
                     collections=['cop-dem-glo-30'],
                     datetime=None,
                     max_items=10000)
    print(len(items))
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    s = tt()
    input_dicts = [{'item': item,
                    'save_name': 'elevation',
                    'save_dir': save_dir,
                    'force_download': force_download,
                    'num_items': len(items),
                    'start_time': s,
                    'print_progress': print_progress
                    } for item in items]
    my_pool(
        num_proc=num_proc, input_list=input_dicts, func=download_and_save_single_file,
        time_out=120, terminate_on_error=False
    )

def download_bbox_list_elevation_data(bbox_list: list[bboxtype],
                                 save_dir: Path = ppaths.country_data/'elevation',
                                 num_proc: int = 2,
                                 force_download: bool = False,
                                 print_progress: bool = True
                                 ) -> None:
    """
    Downloads elevation data for the entered bbox, and saves the data to
    model_data/country/country_elevation/country_elevation_parts
    """
    items = []
    for bbox in bbox_list:
        items.extend(
            query_pc(bbox=bbox, collections=['cop-dem-glo-30'], datetime=None, max_items=10000)
        )
    print(len(items))
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    s = tt()
    input_dicts = [{'item': item,
                    'save_name': 'elevation',
                    'save_dir': save_dir,
                    'force_download': force_download,
                    'num_items': len(items),
                    'start_time': s,
                    'print_progress': print_progress
                    } for item in items]
    my_pool(
        num_proc=num_proc, input_list=input_dicts, func=download_and_save_single_file,
        time_out=120, terminate_on_error=False
    )

def download_country_elevation_data(country: str,
                                    num_proc: int = 4,
                                    print_progress: bool = True,
                                    force_download: bool = False
                                    ):
    """
    Downloads elevation data for the entered country and saves it to
    model_data/country/country_elevation/country_elevation_parts.

    Parameters
    ----------
    country
    print_progress
    """
    bbox = get_country_bounding_box(country_name=country)
    bbox = (max(bbox[0] - .15, -180), max(bbox[1] - .15, -90), min(bbox[2] + .15, 180), min(bbox[3] + .15, 90))
    download_bbox_elevation_data(
        bbox=bbox, num_proc=num_proc, force_download=force_download, print_progress=print_progress
    )


def download_polygon_list_elevation_data(
        polygon_list: list[shapely.Polygon], num_proc: int = 4, save_dir: Path = ppaths.country_data/'elevation',
        print_progress: bool = True, force_download: bool = False
):
    bbox_list = [tuple(polygon.bounds) for polygon in polygon_list]
    download_bbox_list_elevation_data(
        bbox_list=bbox_list, num_proc=num_proc, force_download=force_download,
        print_progress=print_progress, save_dir=save_dir
    )


def get_series_polygon_list(series: gpd.GeoSeries):
    polygon_list = []
    for polygon in series:
        if hasattr(polygon, 'geoms'):
            polygon_list.extend(polygon.geoms)
        else:
            polygon_list.append(polygon)
    return polygon_list

# def get_admin_gdf_from_country_name(country_name: str, admin_level: int):
#     shapefile_path = ppaths.country_lookup_data/'shapefiles'
#     alpha_3 = get_alpha_3_code_from_country_name(country_name)
#     dir_path = shapefile_path/f'gadm41_{alpha_3}_shp'
#     file_path = dir_path/f'gadm41_{alpha_3}_{admin_level}.shp'
#     df = gpd.read_file(file_path)
#     df['name'] = df[f'NAME_{admin_level}']
#     return df

if __name__ == '__main__':


    # country_list = [
    #     'jammu-kashmir', 'aksai chin', 'arunachal pradesh'
    # ]
    gdf = gpd.read_parquet(ppaths.country_lookup_data/'world_boundaries.parquet')
    already_downloaded = shapely.unary_union(gpd.read_parquet(ppaths.country_data/'elevation/elevation.parquet').geometry)
    world_geom = shapely.unary_union(gdf.geometry)
    remaining = world_geom.difference(already_downloaded)
    poly_list = list(remaining.geoms)
    print(len(poly_list))
    # series = gpd.GeoSeries(poly_list)
    # series.plot()
    # for country in country_list:
    #     poly_list.append(gdf[gdf.name.str.lower() == country].reset_index(drop=True).geometry[0])
    download_polygon_list_elevation_data(poly_list)


    #
    #
    # gdf = get_admin_gdf_from_country_name('united_states_of_america', admin_level=1)
    # gdf = gdf[gdf.NAME_1.isin(['Hawaii'])]
    # print(gdf)
    # poly_list = get_series_polygon_list(gdf.geometry)
    # print(len(poly_list))
    # download_polygon_list_elevation_data(poly_list)
    #
    # download_bbox_elevation_data(
    #     bbox = [-175, 51, -120, 74], save_dir=ppaths.country_data/'elevation', num_proc=4,
    # )

    # country_list = ['Greenland']
    #
    # for country in country_list:
    #     print(country)
    #     download_country_elevation_data(country, num_proc=8)