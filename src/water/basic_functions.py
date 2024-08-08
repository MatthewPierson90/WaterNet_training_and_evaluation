import pickle
import warnings

import pandas as pd
import geopandas as gpd
from time import perf_counter as tt
from time import sleep
import sys
from pathlib import Path
import datetime as dt
import pickle as pkl
import json
import yaml
import numpy as np
from shutil import rmtree
import shapely
from rasterio import CRS as rio_crs
import rioxarray as rxr
import xarray as xr
from pyproj import CRS as pyproj_crs, Geod
from shapely.geometry import (Polygon, MultiPolygon, Point, MultiPoint,
                              LineString, MultiLineString, LinearRing)
from typing import Union, Tuple, Iterable, Callable
import requests
import zipfile
import py7zr
import os
from multiprocessing import Process
import psutil


Iterable = Iterable
Callable = Callable
numeric = Union[int, float]
pathlike = Union[str, Path]
dflike = Union[pd.DataFrame, pd.Series, gpd.GeoDataFrame, gpd.GeoSeries]
bboxtype = Tuple[numeric, numeric, numeric, numeric]
crstype = Union[rio_crs, pyproj_crs, str]
geometrylike = Union[
    Polygon, MultiPolygon, Point, MultiPoint, LineString, MultiLineString, LinearRing
]


def save_yaml(file_name: pathlike,
              obj
              ) -> None:
    with open(f'{file_name}', 'w') as file:
        yaml.dump(obj, file)


def open_yaml(file_name: pathlike):
    with open(f'{file_name}', 'r') as file:
        obj = yaml.safe_load(file)
    return obj


package_path = Path(__file__).parent
src_path = package_path.parent
# base_path = bridges_package_path.parent
if (src_path.parent/'data').exists():
    base_path = src_path.parent
elif (Path(sys.path[0])/'data').exists():
    base_path = Path(sys.path[0])
elif (Path(sys.path[0]).parent/'data').exists():
    base_path = Path(sys.path[0]).parent
elif (Path(os.getcwd())/'data').exists():
    base_path = Path(os.getcwd())
elif (Path(os.getcwd()).parent/'data').exists():
    base_path = Path(os.getcwd()).parent
elif (Path(os.getcwd()).parent.parent/'data').exists():
    base_path = Path(os.getcwd()).parent.parent
else:
    raise Exception('Can\'t find data directory... '
                    'A directory named "data" must exist'
                    'in the same directory as this script '
                    'or in this script\'s parent directory'
                    f'Checked:'
                    f'{os.getcwd()}'
                    f'{sys.path[0]}'
                    )


class Proj_paths:
    """
    paths for the project
    """
    
    def __init__(self,
                 base: Path = base_path
                 ) -> None:
        self.base_path = base
        self.configuration_files = self.base_path/'configuration_files'
        self._path_config = {}
        if (self.configuration_files/'path_configuration.yaml').exists():
            self._path_config = open_yaml(self.configuration_files/'path_configuration.yaml')
            if self._path_config is None:
                self._path_config = {}
        self.data = base/'data'
        self.waterway = self.add_directory('waterway_data', self.data)
        self.country_lookup_data = self.add_directory('country_lookup_data', self.data)
        self.country_data = self.add_directory('country_data', self.waterway)
        self.storage_data = self.add_directory('waterway_storage', self.waterway)
    
    def add_directory(self, directory_name: str, directory_parent: Path):
        if directory_name not in self._path_config:
            directory_path = directory_parent/directory_name
        else:
            directory_path = Path(self._path_config[directory_name])
        if not directory_path.exists() and not directory_path.is_symlink():
            directory_path.mkdir()
        return directory_path


ppaths = Proj_paths()


class CountryCodeException(Exception):
    @staticmethod
    def no_info(country_name):
        return f'I have no information on {country_name}'
    
    @staticmethod
    def not_unique(country_name, possible_names):
        return_str = f'{country_name} is not unique, found the following:\n'
        return_str += f'  name: alpha_2_code\n'
        for n in range(len(possible_names)):
            name = possible_names.loc[n, 'country_name']
            code = possible_names.loc[n, 'alpha_2_code']
            return_str += f'  {name}: {code}\n'
        return return_str


def get_country_bbox_from_alpha_2_code(alpha_2_code: str
                                       ) -> bboxtype:
    """
    returns the bounding box for the country with the entered alpha_2_code
    Parameters
    ----------
    alpha_2_code - str

    Returns
    -------
    (min_lon, min_lat, max_lon, max_lat) - (float, float, float, float)
    """
    file_path = ppaths.country_lookup_data/'country_information.parquet'
    df = pd.read_parquet(file_path)
    alpha_2_code = alpha_2_code.lower()
    country_bounding_box = df.loc[alpha_2_code, 'boundingBox']
    min_lon = country_bounding_box['sw']['lon']
    min_lat = country_bounding_box['sw']['lat']
    max_lon = country_bounding_box['ne']['lon']
    max_lat = country_bounding_box['ne']['lat']
    return min_lon, min_lat, max_lon, max_lat


def get_alpha_2_code_from_country_name(country_name: str
                                       ) -> str:
    """
    returns the corresponding alpha 2 code for the entered country_name
    Parameters
    ----------
    country_name - str

    Returns
    -------
    alpha_2_code
    """
    # codes = pd.read_csv(ppaths.country_lookup_data/'country_codes.csv', delimiter='\t')
    codes = pd.read_parquet(ppaths.country_lookup_data/'country_information.parquet')

    codes.country_name = codes.country_name.str.lower()
    country_name = country_name.lower().replace(' ', '_')
    country_codes = codes[codes.country_name == country_name].reset_index(drop=True)
    if len(country_codes) == 0:
        country_codes = codes[codes.country_name.str.contains(country_name)].reset_index(drop=True)
    if len(country_codes) == 1:
        return country_codes.loc[0, 'alpha_2_code']
    elif len(country_codes) == 0:
        raise CountryCodeException(CountryCodeException.no_info(country_name))
    else:
        raise CountryCodeException(CountryCodeException.not_unique(country_name, country_codes))


def get_alpha_3_code_from_country_name(country_name: str
                                       ) -> str:
    """
    returns the corresponding alpha 2 code for the entered country_name
    Parameters
    ----------
    country_name - str

    Returns
    -------
    alpha_2_code
    """

    # codes = pd.read_csv(ppaths.country_lookup_data/'country_codes.csv', delimiter='\t')
    codes = pd.read_parquet(ppaths.country_lookup_data/'country_information.parquet')
    codes.country_name = codes.country_name.str.lower()
    country_name = country_name.lower().replace(' ', '_')
    country_codes = codes[codes.country_name == country_name].reset_index(drop=True)
    if len(country_codes) == 0:
        country_codes = codes[codes.country_name.str.contains(country_name)].reset_index(drop=True)
    if len(country_codes) == 1:
        return country_codes.loc[0, 'alpha_3_code']
    elif len(country_codes) == 0:
        raise CountryCodeException(CountryCodeException.no_info(country_name))
    else:
        raise CountryCodeException(CountryCodeException.not_unique(country_name, country_codes))


def get_alpha_2_code_from_alpha_3_code(alpha_3_code: str) -> str:
    """
    returns the corresponding alpha 2 code for the entered alpha 3 code
    Parameters
    ----------
    alpha_3_code- str

    Returns
    -------
    alpha_2_code
    """
    codes = pd.read_csv(ppaths.country_lookup_data/'country_codes.csv')
    codes.alpha_3_code = codes.alpha_3_code.str.lower()
    alpha_3_code = alpha_3_code.lower()
    country_codes = codes[codes.alpha_3_code.str.contains(alpha_3_code)].reset_index(drop=True)
    if len(country_codes) == 1:
        return country_codes.loc[0, 'alpha_3_code']
    elif len(country_codes) == 0:
        raise CountryCodeException(CountryCodeException.no_info(alpha_3_code))
    else:
        raise CountryCodeException(CountryCodeException.not_unique(alpha_3_code, country_codes))


def open_gdf(file_path: Path):
    if 'parquet' in file_path.suffix.lower():
        return gpd.read_parquet(path=file_path)
    else:
        return gpd.read_file(file_path)


def get_multipolygon_max_polygon(multipolygon: shapely.MultiPolygon or shapely.Polygon):
    max_area = 0
    polygon = multipolygon
    if hasattr(multipolygon, 'geoms'):
        polygon_geoms = list(multipolygon.geoms)
        for geom in polygon_geoms:
            if geom.area > max_area:
                polygon = geom
                max_area = geom.area
    return polygon


def get_polygons_by_alpha3(alpha_3_code: str, only_mainland: bool=True):
    gdf = gpd.read_parquet(ppaths.country_lookup_data/'world_boundaries.parquet')
    polygon = gdf[gdf.iso3 == alpha_3_code.upper()].reset_index(drop=True)['geometry'][0]
    if only_mainland:
        polygon = get_multipolygon_max_polygon(polygon)
    return polygon


def get_country_polygon(country, only_mainland=False):
    alpha3 = get_alpha_3_code_from_country_name(country)
    return get_polygons_by_alpha3(alpha3, only_mainland)


def get_country_bridges(country):
    country_polygon = get_country_polygon(country)
    bridges = gpd.read_parquet(ppaths.country_data/'bridge_locations.parquet')
    bridges = bridges[bridges.intersects(country_polygon)].reset_index(drop=True)
    return bridges

def get_country_bounding_box(country_name: str=None,
                             alpha_2_code: str=None,
                             alpha_3_code: str=None
                             ) -> bboxtype:
    """
    returns country bounding box, only requires one of country_name, alpha_2_code, or alpha_3_code.
    If country_name or alpha_3_code are entered,
    then they will first be converted to the country's alpha_2_code.

    Parameters
    ----------
    country_name - str
    alpha_2_code - str
    alpha_3_code - str

    Returns
    -------

    """
    try:
        if alpha_2_code is not None:
            min_lon, min_lat, max_lon, max_lat = get_country_bbox_from_alpha_2_code(alpha_2_code)
        else:
            if country_name is not None:
                alpha_2_code = get_alpha_2_code_from_country_name(country_name)
            elif alpha_3_code is not None:
                alpha_2_code = get_alpha_2_code_from_alpha_3_code(alpha_3_code)
            else:
                raise Exception(
                        print('One of country_name, alpha_2_code, or alpha_3_code must have an input')
                )
            min_lon, min_lat, max_lon, max_lat = get_country_bbox_from_alpha_2_code(alpha_2_code)
    except KeyError:
        polygon = get_country_polygon(country=country_name)
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
    return min_lon - .005, min_lat - .005, max_lon + .005, max_lat + .005


def check_hu4_exists(hu4_index: int):
    return (ppaths.waterway/f'hu4_hull/hu4_{hu4_index:04d}.parquet').exists()


def get_hu4_hull_gdf(hu4_index: int):
    gdf = gpd.read_parquet(ppaths.waterway/f'hu4_hull/hu4_{hu4_index:04d}.parquet')
    return gdf

def get_hu4_hull_polygon(hu4_index: int):
    gdf = get_hu4_hull_gdf(hu4_index).reset_index(drop=True)
    polygon = gdf.loc[0, 'geometry']
    return polygon

def get_hu4_gdf(hu4_index: int):
    gdf = gpd.read_parquet(ppaths.waterway/f'hu4_parquet/hu4_{hu4_index:04d}.parquet')
    return gdf

def file_name_to_bbox(file_name: str, buffer: float) -> list:
    bbox = file_name.split('.tif')[0].split('_')[1:]
    bbox = [float(val) - buffer*(-1)**(ind//2) for ind, val in enumerate(bbox)]
    return bbox


def name_to_box(file_name: str, buffer: float = 0.) -> shapely.box:
    bbox = file_name_to_bbox(file_name, buffer=buffer)
    return shapely.box(*bbox)

def save_json(file_name: pathlike,
              obj
              ) -> None:
    with open(f'{file_name}', 'w') as file:
        json.dump(obj, file)


def open_json(file_name: pathlike):
    with open(f'{file_name}', 'r') as file:
        obj = json.load(file)
    return obj


def save_yaml(file_name: pathlike,
              obj
              ) -> None:
    with open(f'{file_name}', 'w') as file:
        yaml.dump(obj, file)


def open_yaml(file_name: pathlike):
    with open(f'{file_name}', 'r') as file:
        obj = yaml.load(file, yaml.Loader)
    return obj


def get_test_file_names():
    with open(ppaths.waterway/'test_file_list.txt') as f:
        files = f.read().split('\n')[:-1]
    return files


def get_val_file_names():
    with open(ppaths.waterway/'val_file_list.txt') as f:
        files = f.read().split('\n')[:-1]
    return files


def delete_directory_contents(directory_path: Path):
    """
    Deletes all content in entered directory.
    DO NOT USE THIS FUNCTION IF YOU DON'T KNOW WHAT YOU ARE DOING!
    NEVER EXPOSE THIS FUNCTION TO A USER (AND MAKE SURE ANY PATH
    PASSED TO THIS FUNCTION IS NOT EXPOSED TO A USER).
    Parameters
    ----------
    directory_path
    """
    if directory_path.exists() and directory_path.is_dir():
        rmtree(directory_path)
        directory_path.mkdir()
    elif not directory_path.exists():
        warnings.warn(f'Directory {directory_path} does not exist.', category=RuntimeWarning)
    else:
        warnings.warn(f'{directory_path} is not a directory.', category=RuntimeWarning)


def resuffix_directory_and_make_new(directory_path: Path):
    """
    Adds a suffix to the entered directory, then makes an empty directory with that name.
    DO NOT USE THIS FUNCTION IF YOU DON'T KNOW WHAT YOU ARE DOING!
    NEVER EXPOSE THIS FUNCTION TO A USER (AND MAKE SURE ANY PATH
    PASSED TO THIS FUNCTION IS NOT EXPOSED TO A USER).

    Parameters
    ----------
    directory_path
    """
    parent_dir = directory_path.parent
    directory_name = directory_path.name
    suffix = len(list(parent_dir.glob(f'{directory_name}')))
    new_path = parent_dir/f'{directory_name}_{suffix}'
    while new_path.exists():
        suffix += 1
        new_path = parent_dir/f'{directory_name}_{suffix}'
    directory_path.rename(new_path)
    directory_path.mkdir()

def resuffix_file(file_path: Path):
    """
    Adds a suffix to the entered directory, then makes an empty directory with that name.
    DO NOT USE THIS FUNCTION IF YOU DON'T KNOW WHAT YOU ARE DOING!
    NEVER EXPOSE THIS FUNCTION TO A USER (AND MAKE SURE ANY PATH
    PASSED TO THIS FUNCTION IS NOT EXPOSED TO A USER).

    Parameters
    ----------
    directory_path
    """
    parent_dir = file_path.parent
    full_file_name = file_path.name
    file_type = file_path.suffix
    file_name = full_file_name.split(file_type)[0]
    suffix = len(list(parent_dir.glob(f'{file_name}')))
    new_path = parent_dir/f'{file_name}_{suffix}{file_type}'
    while new_path.exists():
        suffix += 1
        new_path = parent_dir/f'{file_name}_{suffix}{file_type}'
    file_path.rename(new_path)
    return new_path

def save_pickle(file_name: pathlike,
                obj
                ) -> None:
    """
    Saves obj as pkl file

    Parameters
    ----------
    file_name: str or path. The path where the pickled object will be saved.
    obj: python object to pickle

    Returns
    -------
    None
    """
    with open(f'{file_name}', 'wb') as file:
        pkl.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def open_pickle(file_name: pathlike
                ):
    """
    opens pkl file

    Parameters
    ----------
    file_name: str or Path, path of pkl file.

    Returns
    -------
    object that was pickled
    """
    with open(file_name, 'rb') as pkl_file:
        obj = pkl.load(pkl_file)
    return obj


def get_current_time() -> str:
    """
    Gets the current time
    """
    current_time = dt.datetime.now().time()
    time_string = f'{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}'
    return time_string


def print_start_time() -> str:
    """
    Prints the start time and returns the associated time.perf_counter
    """
    print(get_current_time())


def time_elapsed(start: numeric,
                 spaces: int=0
                 ) -> str:
    """
    Prints the time elapsed from the entered start time.

    Parameters
    ----------
    start - float, start time obtained from time.perf_counter
    """
    end = tt()
    if end - start < 1:
        statement = ' '*spaces+f'Time Elapsed: {end-start:.6f}s'
        print(statement)
    elif end - start < 60:
        statement = ' '*spaces+f'Time Elapsed: {end-start:.2f}s'
        print(statement)
    elif end - start < 3600:
        mins = int((end-start)/60)
        sec = (end-start) % 60
        statement = ' '*spaces+f'Time Elapsed: {mins}m, {sec:.2f}s'
        print(statement)
    else:
        hrs = int((end-start)/3600)
        mins = int(((end-start) % 3600)/60)
        sec = (end-start) % 60
        statement = ' '*spaces+f'Time Elapsed: {hrs}h, {mins}m, {sec:.2f}s'
        print(statement)
    return statement


def wait_n_seconds(time: float):
    s = tt()
    while tt() - s < time:
        continue


def printdf(df: dflike,
            head: int = 5,
            start_index: int=0,
            head_tail: str='head',
            include_geometry: bool=False
            ) -> None:
    """
    prints the head of a DataFrame

    Parameters
    ----------
    df - pd.DataFrame, data frame (or series) to print
    head - int, number of rows to print

    Returns
    -------
    None
    """
    try:
        cols = df.columns if include_geometry else [col for col in df.columns if col != 'geometry']
        if head_tail == 'head':
            print(df[cols].iloc[start_index:].head(head).to_string())
        else:
            print(df[cols].iloc[start_index:].tail(head).to_string())
    except AttributeError:
        if head_tail == 'head':
            print(df[cols].iloc[start_index:].head(head).to_string())
        else:
            print(df[cols].iloc[start_index:].tail(head).to_string())

def get_bbox_x_meter_resolution(resolution_in_meters,
                                x_min=None,
                                y_min=None,
                                x_max=None,
                                y_max=None,
                                bbox=None):
    """
    Finds an approximation of the entered resolution_in_meters in
    degrees (wgs84) using the middle latitude value.

    Parameters
    ----------
    bbox
    resolution_in_meters

    Returns
    -------
    resolution in degrees
    """
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
    if y_min is None:
        raise Exception('Either y_min, y_max must be entered or the bounding box must be entered')
    if x_min is None:
        x_min = 0
    avg_lat = (y_max + y_min) / 2
    geod = Geod(ellps='WGS84')
    _, lat1, _ = geod.fwd(lons=[x_min], lats=[avg_lat], az=0, dist=resolution_in_meters)
    resolution = abs(lat1[0] - avg_lat)
    return resolution


def coordinates_to_row_col(x: numeric,
                           y: numeric,
                           x_min: numeric,
                           y_max: numeric,
                           resolution: numeric=None,
                           x_resolution: float=None,
                           y_resolution: float=None
                           ) -> (int, int):
    """
    Finds the row, col of the grid cell that (x, y) falls in.

    Parameters
    ----------
    x: The x-coordinate to be found.
    y: The y-coordinate to be found.
    x_min: The minimum x-value (ie the west coordinate of the grid's bounding box)
    y_max: The maximum y-value (ie the north coordinate of the grid's bounding box)
    resolution: The resolution of the grid

    Returns
    -------
    row, col: the row and column of the grid cell.
    """
    if resolution is not None:
        col = int(np.floor((x - x_min) / resolution))
        row = int(np.floor((y_max - y) / resolution))
    elif x_resolution is not None and y_resolution is not None:
        col = int(np.floor((x - x_min) / x_resolution))
        row = int(np.floor((y_max - y) / y_resolution))
    else:
        raise Exception('Either resolution or x_resolution and y_resolution must not be None')
    return row, col


def coordinate_list_to_row_col_array(coordinate_list: list,
                                     x_min: float,
                                     y_max: float,
                                     resolution: float=None,
                                     x_resolution: float=None,
                                     y_resolution: float=None
                                     ) -> np.ndarray:
    """
    Finds the row, col for each of the (x, y) coordinates in coordinate_list.
    Returns these row, col pairs in a np.ndarray.

    Parameters
    ----------
    coordinate_list: The (x,y)-coordinates to be found.
    x_min: The minimum x-value (ie the west coordinate of the grid's bounding box)
    y_max: The maximum y-value (ie the north coordinate of the grid's bounding box)
    resolution: The resolution of the grid

    Returns
    -------
    row_col_array: The corresponding row, col values in a np.ndarray.
    """
    coordinate_array = np.array(coordinate_list, dtype=float)
    x = coordinate_array[:, 0]
    y = coordinate_array[:, 1]
    row_col_array = np.zeros(coordinate_array.shape)
    if resolution is not None:
        row_col_array[:, 0] = np.floor((y_max - y) / resolution)
        row_col_array[:, 1] = np.floor((x - x_min) / resolution)
    elif x_resolution is not None and y_resolution is not None:
        row_col_array[:, 0] = np.floor((y_max - y) / y_resolution)
        row_col_array[:, 1] = np.floor((x - x_min) / x_resolution)
    else:
        raise Exception('Either resolution or x_resolution and y_resolution must not be None')
    row_col_array = row_col_array.astype(int)
    return row_col_array


def row_col_array_to_midpoint_coordinates(row_col_array: Iterable[Iterable[int]],
                                          x_min: numeric,
                                          y_max: numeric,
                                          resolution: numeric = None,
                                          x_resolution: numeric = None,
                                          y_resolution: numeric = None
                                          )-> np.array:
    """
    Finds coordinates of the midpoints for (row,col) values in row_col_list
    """
    if type(row_col_array) != np.ndarray:
        row_col_array = np.array(row_col_array)
    x_y_array = np.zeros(row_col_array.shape)
    if x_resolution is not None:
        if y_resolution < 0:
            y_resolution = -y_resolution
        x_y_array[:, 0] = x_min + x_resolution*(row_col_array[:, 1] + .5)
        x_y_array[:, 1] = y_max - y_resolution*(row_col_array[:, 0] + .5)
    else:
        x_y_array[:, 0] = x_min + resolution*(row_col_array[:, 1] + .5)
        x_y_array[:, 1] = y_max - resolution*(row_col_array[:, 0] + .5)
    return x_y_array


def row_col_bbox_to_xy_bbox(rc_bbox, x_min, y_max, x_resolution, y_resolution):
    y_resolution = np.abs(y_resolution)
    cm, rm, cM, rM = rc_bbox
    bbox_x_min = x_min + x_resolution*cm
    bbox_y_min = y_max - y_resolution*rM
    bbox_x_max = x_min + x_resolution*cM
    bbox_y_max = y_max - y_resolution*rm
    return bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max

def make_elevation_file_array_and_file_paths(
        elevation_dir: Path = ppaths.country_data/'elevation'
) -> ('np.array', list):
    elevations_files = list(elevation_dir.glob('*.tif'))
    elevation_array = np.zeros((len(elevations_files), 4))
    for ind, file in enumerate(elevations_files):
        name = file.name
        split = name.split('elevation_')[-1].split('.tif')[0].split('_')
        for i, val in enumerate(split):
            elevation_array[ind, i] = float(val)
    return elevation_array, elevations_files


def split_coordinates_by_elevation_file(coordinate_array: 'np.array',
                                        elevation_array: 'np.array'
                                        ) -> dict:
    split_coordinate_index_lists = {}
    for ind, (x_min, y_min, x_max, y_max) in enumerate(elevation_array):
        indices = np.where((x_min <= coordinate_array[:, 0])
                           & (coordinate_array[:, 0] < x_max)
                           & (y_min <= coordinate_array[:, 1])
                           & (coordinate_array[:, 1] < y_max))[0]
        if len(indices) > 0:
            split_coordinate_index_lists[ind] = indices
    return split_coordinate_index_lists


def get_elevation_from_file(coordinate_array: 'np.array',
                            elevation_file: Path
                            ) -> 'np.array':
    with rxr.open_rasterio(elevation_file) as rio_f:
        x = xr.DataArray(coordinate_array[:, 0], dims='points')
        y = xr.DataArray(coordinate_array[:, 1], dims='points')
        elevations = rio_f[0].sel(x=x, y=y, method='nearest').values.astype(float)
    return elevations


def coordinate_elevation_lookup(coordinate_list: list,
                                elevation_dir: Path = ppaths.country_data/'elevation',
                                ) -> 'np.array':
    coordinate_array = np.array(coordinate_list)
    to_return = np.zeros(len(coordinate_list))
    elevation_array, elevation_paths = make_elevation_file_array_and_file_paths(elevation_dir)
    coordinate_index_lists = split_coordinates_by_elevation_file(
        coordinate_array=coordinate_array, elevation_array=elevation_array
    )
    for index in coordinate_index_lists:
        indices = coordinate_index_lists[index]
        file_path = elevation_paths[index]
        to_return[indices] = get_elevation_from_file(coordinate_array[indices], file_path)
    return to_return


def extract_all_from_file(file_path: Path,
                          delete_after_extraction: bool=False
                          ):
    """
    Checks if file is of type .7z or zip, if it is, it will extract the contents.

    Parameters
    ----------
    file_path - path
    delete_after_extraction - set to true to delete the zip file after extraction
    Returns
    -------
    Path - Path to the newly extracted directory (or the original file path),
    bool - True if file was extracted else False.
    """
    path_to_return = file_path
    true_false = False
    was_zip = False
    was_7z = False
    load_func = None
    if py7zr.is_7zfile(file_path) and ('.7z' in file_path.name):
        load_func = py7zr.SevenZipFile
        was_zip = True
        was_7z = True
    elif zipfile.is_zipfile(file_path) and ('.zip' in file_path.name):
        load_func = zipfile.ZipFile
        was_zip = True

    if was_zip:
        with load_func(file_path, mode='r') as z:
            new_path = file_path.parent/f'{file_path.name.split(file_path.suffix)[0]}'
            path_to_return = new_path
            if not new_path.exists():
                was_base = True
                if not was_7z:
                    for name in z.namelist():
                        if new_path.name not in name:
                            was_base = False
                            break
                else:
                    was_base = False
                if was_base:
                    z.extractall(path=new_path.parent)
                else:
                    new_path.mkdir()
                    z.extractall(path=new_path)
                true_false = True
        if delete_after_extraction:
            os.remove(file_path)
    return path_to_return, true_false


def extract_all_recursive(dir_path: Path,
                          delete_after_extraction: bool = False
                          ):
    """
    Recursively extracts zip, 7z files from a directory.

    Parameters
    ----------
    dir_path - path of the top directory
    delete_after_extraction - Set to true to delete the zip file after it is extracted
    Returns
    -------
    None
    """
    if not dir_path.is_dir():
        dir_path, _ = extract_all_from_file(file_path=dir_path,
                                            delete_after_extraction=delete_after_extraction)
    children = list(dir_path.glob('*'))
    ind = 0
    while ind < len(children):
        child = children[ind]
        if child.is_dir():
            extract_all_recursive(child)
        else:
            new_child, was_extracted = extract_all_from_file(
                    file_path=child, delete_after_extraction=delete_after_extraction
            )
            if was_extracted:
                children.append(new_child)
        ind += 1


def move_zip_file(file_path: Path):
    """
    Moves zip file in data to data/zip_files
    Parameters
    ----------
    file_path - path

    """
    if '.7z' in file_path.name or '.zip' in file_path.name:
        new_path = Path(str(file_path).replace('/data/', '/data/zip_files/'))
        new_parent = new_path.parent
        if not new_parent.exists():
            new_parent.mkdir(parents=True)
        file_path.rename(new_path)


def move_zip_files_recursive(dir_path: Path):
    """
    Moves zip files in the subdirectories of data to data/zip_files

    Parameters
    ----------
    dir_path - path
    """
    data_children = dir_path.glob('*')
    for child_path in data_children:
        if 'zip_files' not in child_path.name:
            if child_path.is_dir():
                move_zip_files_recursive(child_path)
            else:
                move_zip_file(child_path)


def single_download(url, save_path, extract_zip_files: bool = False, delete_zip_files_after_extraction: bool = False):
    try:
        with requests.get(url, timeout=5, stream=True) as response:
            if response.ok:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1*10**8):
                        f.write(chunk)
                if extract_zip_files:
                    extract_all_recursive(
                            dir_path=save_path,
                            delete_after_extraction=delete_zip_files_after_extraction
                    )
                return True
            else:
                print(f'Failed to download {url}')
                return False
    except requests.exceptions.ConnectionError:
        sleep(10)
        single_download(
            url=url, save_path=save_path, extract_zip_files=extract_zip_files,
            delete_zip_files_after_extraction=delete_zip_files_after_extraction
        )


def multi_download(url_list: list,
                   save_path_list: list,
                   extract_zip_files: bool=False,
                   delete_zip_files_after_extraction: bool=True,
                   num_proc: int = -1):
    input_dicts = [{'url': url,
                    'save_path': save_path,
                    'extract_zip_files': extract_zip_files,
                    'delete_zip_files_after_extraction': delete_zip_files_after_extraction,
                    } for url, save_path in zip(url_list, save_path_list)]
    my_pool(num_proc=num_proc, input_list=input_dicts, func=single_download, time_out=3600, use_kwargs=True)


def add_new_process(func, inputs, process_dict, use_kwargs, name=None):
    # print(inputs)
    if use_kwargs:
        p = Process(target=func, kwargs=inputs, name=name)
    else:
        p = Process(target=func, args=(inputs, ), name=name)
    p.start()
    # process_dict[p.pid] = (p, tt(), inputs)
    process_dict[p.pid] = {
        'process': p, 'start_time': tt(), 'inputs': inputs, 'cpu_time': 0, 'cpu_time_delta': 0
    }
    return process_dict


def terminate_all(process_dict):
    for process_info in process_dict.values():
        p = process_info['process']
        p.terminate()
        p.join()
        p.close()
    raise Exception('One of the processes failed')


def remove_process(completed_pids, process_dict, terminate_on_error):
    for pid in completed_pids:
        try:
            p = process_dict[pid]['process']
            p.join()
            exitcode = p.exitcode
            p.close()
            process_dict.pop(pid)
            if exitcode == 1 and terminate_on_error:
                terminate_all(process_dict)
        except:
            print(pid, process_dict.keys())
    return process_dict


def check_for_completed_processes(process_dict: dict, time_out: float, input_list: list, time_delta_margin: float = 1):
    to_return = []
    num_timed_out = 0
    for pid, process_dict in process_dict.items():
        p = process_dict['process']
        new_time = cpu_time = process_dict['cpu_time']
        start_time = process_dict['start_time']
        time_delta = 0
        if not p.is_alive():
            to_return.append(pid)
        else:
            new_time = sum(psutil.Process(pid).cpu_times())
            time_delta = new_time - cpu_time
        if time_delta > time_delta_margin:
            process_dict['cpu_time'] = new_time
            process_dict['start_time'] = tt()
        else:
            if tt() - start_time > time_out:
                print(f'{pid} timed out')
                inputs = process_dict['inputs']
                p.terminate()
                to_return.append(pid)
                input_list.append(inputs)
                num_timed_out += 1
    return to_return, input_list, num_timed_out


def my_pool(num_proc, func,
            input_list,
            use_kwargs=False,
            time_out=None, sleep_time=.1,
            terminate_on_error=False,
            print_progress=True,
            name=None):
    max_proc = os.cpu_count()
    if time_out is None:
        time_out = np.inf
    if num_proc > max_proc:
        num_proc = max_proc
    elif num_proc <= 0:
        num_proc = max_proc + num_proc
    current_input_index = 0
    process_dict = {}
    num_completed = 0
    num_to_complete = len(input_list)
    last_printed = -1
    percentile = max(num_to_complete//100, 1)
    s = tt()
    while True:
        while len(process_dict) < num_proc and current_input_index < len(input_list):
            process_dict = add_new_process(
                func=func, inputs=input_list[current_input_index], process_dict=process_dict, use_kwargs=use_kwargs,
                name=f'{name}_{current_input_index}' if name is not None else None
            )
            current_input_index += 1
        completed_pids, input_list, num_timed_out = check_for_completed_processes(process_dict,
                                                                                  time_out,
                                                                                  input_list)
        num_completed += len(completed_pids) - num_timed_out
        if print_progress:
            if num_completed//percentile > last_printed:
                last_printed = num_completed//percentile
                if num_to_complete == 0:
                    num_to_complete = 1
                percent = num_completed/num_to_complete
                print(f'Completed {percent:.2%} ({num_completed}/{num_to_complete})')
                time_elapsed(s, 2)
        process_dict = remove_process(completed_pids=completed_pids, process_dict=process_dict,
                                      terminate_on_error=terminate_on_error)
        if len(process_dict) == 0 and current_input_index >= len(input_list):
            break
        if len(process_dict) == num_proc:
            if sleep_time > 0:
                sleep(sleep_time)


def make_directory_gdf(dir_path: Path, use_name: bool = True):
    import rasterio as rio
    if not use_name:
        from pyproj import Transformer
        from pyproj.crs import CRS
    dir_name = dir_path.name
    parquet_path = dir_path/f'{dir_name}.parquet'
    if not parquet_path.exists():
        file_paths = list(dir_path.glob('*.tif'))
        dir_gdf = {'file_name': [], 'geometry': []}
        for file in file_paths:
            dir_gdf['file_name'].append(file.name)
            if use_name:
                dir_gdf['geometry'].append(name_to_box(file.name, 0))
            else:
                with rio.open(file) as rio_f:
                    crs = rio_f.crs
                    crs_4326 = CRS.from_epsg(4326)
                    bbox = tuple(rio_f.bounds)
                    if crs != crs_4326:
                        transformer = Transformer.from_crs(crs_from=crs, crs_to=crs_4326, always_xy=True)
                        bbox = transformer.transform_bounds(*bbox)
                    box = shapely.box(*bbox)
                dir_gdf['geometry'].append(box)
        dir_gdf = gpd.GeoDataFrame(dir_gdf, crs=4326)
        dir_gdf.to_parquet(parquet_path)
    else:
        dir_gdf = gpd.read_parquet(parquet_path)
        if len(dir_gdf) != len(list(dir_path.glob('*.tif'))):
            os.remove(parquet_path)
            dir_gdf = make_directory_gdf(dir_path, use_name)
    return dir_gdf



if __name__ == '__main__':
    gdf = make_directory_gdf(ppaths.country_data/'model_outputs_zoom_level_6', use_name=False)
    # gdf = make_directory_gdf(ppaths.country_data/'elevation', use_name=True)

    # gdf.to_parquet(ppaths.waterway/'storage_sentinel_4326.parquet')
#     s=tt()
#     func = lambda x: sum([1/n**2 for n in range(1, x)])
#     inputs = 1000000
#     p = Process(target=func, args=(inputs,))
#     p.start()
#     p.terminate()
#     print(p.exitcode)
#     p.join()
#     print(p.exitcode)
#     p.close()
#     time_elapsed(s)
