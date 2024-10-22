import shapely

from water.basic_functions import tt, bboxtype, SharedMemoryPool
from water.paths import ppaths
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
                                 save_dir: Path = ppaths.elevation_data,
                                 num_proc: int = 2,
                                 force_download: bool = False,
                                 print_progress: bool = True
                                 ) -> None:
    """
    Downloads elevation data for the entered bbox, and saves the data to
    model_data/country/country_elevation/country_elevation_parts
    """
    items = query_pc(
        bbox=bbox, collections=['cop-dem-glo-30'], datetime=None, max_items=10000
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
    SharedMemoryPool(
        num_proc=num_proc, input_list=input_dicts, func=download_and_save_single_file,
        time_out=120, terminate_on_error=False
    ).run()


def download_bbox_list_elevation_data(
        bbox_list: list[bboxtype], save_dir: Path = ppaths.elevation_data,
        num_proc: int = 2, force_download: bool = False, print_progress: bool = True
) -> None:
    """
    Downloads elevation data for the entered bbox, and saves the data to
    model_data/country/country_elevation/country_elevation_parts
    """
    items = {}
    for bbox in bbox_list:
        new_items = {
            item.id: item
            for item in query_pc(bbox=bbox, collections=['cop-dem-glo-30'], datetime=None, max_items=10000)
        }
        items.update(new_items)
    items = list(items.values())
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    s = tt()
    input_dicts = [{
        'item': item, 'save_name': 'bbox', 'save_dir': save_dir, 'force_download': force_download,
        'num_items': len(items), 'start_time': s, 'print_progress': print_progress
    } for item in items]
    SharedMemoryPool(
        num_proc=num_proc, input_list=input_dicts, func=download_and_save_single_file,
        time_out=120, terminate_on_error=False
    ).run()


def download_polygon_list_elevation_data(
        polygon_list: list[shapely.Polygon], num_proc: int = 4, save_dir: Path = ppaths.elevation_data,
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
