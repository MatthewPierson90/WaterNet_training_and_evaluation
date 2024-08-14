import pickle
import warnings

import pandas as pd
import geopandas as gpd
from time import perf_counter as tt
from time import sleep
from pathlib import Path
import datetime as dt
import pickle as pkl
import json
import yaml
import numpy as np
from shutil import rmtree
import shapely
from rasterio import CRS as rio_crs
from pyproj import CRS as pyproj_crs
from shapely.geometry import (Polygon, MultiPolygon, Point, MultiPoint,
                              LineString, MultiLineString, LinearRing)
from typing import Union, Tuple, Iterable, Callable
import requests
import zipfile
import py7zr
import os
from multiprocessing import Process
import psutil
from water.paths import ppaths

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


def check_hu4_exists(hu4_index: int):
    return (ppaths.training_data/f'hu4_hull/hu4_{hu4_index:04d}.parquet').exists()


def get_hu4_hull_gdf(hu4_index: int):
    gdf = gpd.read_parquet(ppaths.hu4_hull/f'hu4_{hu4_index:04d}.parquet')
    return gdf


def get_hu4_hull_polygon(hu4_index: int):
    gdf = get_hu4_hull_gdf(hu4_index).reset_index(drop=True)
    polygon = gdf.loc[0, 'geometry']
    return polygon


def get_hu4_gdf(hu4_index: int):
    gdf = gpd.read_parquet(ppaths.hu4_parquet/f'hu4_{hu4_index:04d}.parquet')
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


def get_test_file_names():
    with open(ppaths.training_data/'test_file_list.txt') as f:
        files = f.read().split('\n')[:-1]
    return files


def get_val_file_names():
    with open(ppaths.training_data/'val_file_list.txt') as f:
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
    SharedMemoryPool(
        num_proc=num_proc, input_list=input_dicts, func=single_download, time_out=3600, use_kwargs=True
    ).run()


class SharedMemoryPool:
    def __init__(
            self, func, input_list: list, num_proc: int,
            time_out: float = None,
            sleep_time: float = .1,
            terminate_on_error: bool = False,
            max_memory_usage_percent: float = 75.,
            terminate_memory_usage_percent: float = 90.,
            time_delta_margin: float = 1.,
            use_kwargs: bool = False,
            print_progress: bool = False,
            name: str = None
    ):
        self.func = func
        self.input_list = input_list
        max_proc = os.cpu_count()
        self.time_out = time_out if time_out is not None else np.inf
        if num_proc > max_proc - 1:
            num_proc = max_proc - 1
        elif num_proc <= 0:
            num_proc = min(max(max_proc + num_proc - 1, 1), max_proc - 1)
        self.num_proc = num_proc
        self.current_input_index = 0
        self.process_dict = {}
        self.sleep_time = sleep_time
        self.terminate_on_error = terminate_on_error
        self.max_memory_usage_percent = max_memory_usage_percent
        self.terminate_memory_usage_percent = terminate_memory_usage_percent
        self.time_delta_margin = time_delta_margin
        self.use_kwargs = use_kwargs
        self.to_print_progress = print_progress
        self.name = name
        self.current_input_index = 0
        self.num_completed = 0
        self.num_new_completed = 0
        self.previous_completed = 0
        self.start_time = tt()
        self.num_to_complete = len(self.input_list)

    def has_memory_issues(self):
        current_memory_usage_percent = psutil.virtual_memory().percent
        return current_memory_usage_percent >= self.max_memory_usage_percent

    def has_available_processors(self):
        return len(self.process_dict) < self.num_proc

    def has_more_inputs(self):
        return self.current_input_index < len(self.input_list)

    def get_name(self, current_input_index):
        return f'{self.name}_{current_input_index}' if self.name is not None else None

    def add_new_process(self):
        # print(inputs)
        inputs = self.input_list[self.current_input_index]
        if self.use_kwargs:
            p = Process(target=self.func, kwargs=inputs, name=self.name)
        else:
            p = Process(target=self.func, args=(inputs,), name=self.name)
        p.start()
        self.process_dict[p.pid] = {
            'process': p, 'start_time': tt(), 'inputs': inputs, 'cpu_time': 0, 'cpu_time_delta': 0,
            'init_start_time': tt()
        }
        self.current_input_index += 1

    def fix_memory_usage(self):
        while psutil.virtual_memory().percent > self.terminate_memory_usage_percent:
            pid_to_terminate = list(self.process_dict.keys())[-1]
            newest_start_time = np.inf
            for pid, process_dict in self.process_dict.items():
                if process_dict['init_start_time'] < newest_start_time:
                    pid_to_terminate = pid
                    newest_start_time = process_dict['init_start_time']
            self.terminate_and_restart(pid_to_terminate)

    def check_for_completed_processes_and_timeouts(self):
        self.num_new_completed = 0
        pids = list(self.process_dict)
        for pid in pids:
            process_info_dict = self.process_dict[pid]
            p = process_info_dict['process']
            new_time = cpu_time = process_info_dict['cpu_time']
            start_time = process_info_dict['start_time']
            time_delta = 0
            if not p.is_alive():
                self.remove_process(pid)
                self.num_completed += 1
                self.num_new_completed += 1
            else:
                new_time = sum(psutil.Process(pid).cpu_times())
                time_delta = new_time - cpu_time
            if time_delta > self.time_delta_margin:
                process_info_dict['cpu_time'] = new_time
                process_info_dict['start_time'] = tt()
            else:
                if tt() - start_time > self.time_out:
                    print(f'{pid} timed out')
                    self.terminate_and_restart(pid)

    def terminate_and_restart(self, pid):
        process_info_dict = self.process_dict[pid]
        p = process_info_dict['process']
        p.terminate()
        inputs = process_info_dict['inputs']
        self.input_list.append(inputs)
        self.remove_process(pid)

    def terminate_all(self):
        for process_info_dict in self.process_dict.values():
            q = process_info_dict['process']
            q.terminate()
            q.join()
            q.close()
        raise Exception('One of the processes failed')

    def remove_process(self, pid):
        try:
            p = self.process_dict[pid]['process']
            p.join()
            exitcode = p.exitcode
            p.close()
            self.process_dict.pop(pid)
            if exitcode == 1 and self.terminate_on_error:
                self.terminate_all()
        except KeyError:
            print(pid, self.process_dict.keys())

    def print_progress(self):
        if self.num_new_completed > 0:
            if self.to_print_progress:
                percent_completed = self.num_completed/self.num_to_complete
                ten_percent = (int(100*percent_completed)//10)*10
                if ten_percent > self.previous_completed:
                    print(
                        f'Completed {percent_completed:.2%} ({self.num_completed}/{self.num_to_complete})'
                    )
                    time_elapsed(self.start_time, 2)
                    self.previous_completed = ten_percent

    def run(self):
        try:
            while True:
                while self.has_available_processors() and self.has_more_inputs() and not self.has_memory_issues():
                    self.add_new_process()
                self.check_for_completed_processes_and_timeouts()
                self.fix_memory_usage()
                self.print_progress()
                if len(self.process_dict) == 0 and self.current_input_index >= len(self.input_list):
                    break
                if len(self.process_dict) == self.num_proc or self.has_memory_issues():
                    sleep(self.sleep_time)
        except:
            self.terminate_all()


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
