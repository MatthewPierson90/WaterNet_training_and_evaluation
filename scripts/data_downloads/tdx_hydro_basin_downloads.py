import os
from water.basic_functions import multi_download
from water.paths import ppaths, Path
import geopandas as gpd

africa_indices = [
    1020000010, 1020011530, 1020018110, 1020021940, 1020027430, 1020034170, 1020035180, 1020040190
]
europe_indices = [
    2020000010, 2020003440, 2020018240, 2020024230, 2020033490, 2020041390, 2020057170, 2020065840, 2020071190
]
siberia_indices = [
    3020000010, 3020003790, 3020005240, 3020008670, 3020009320, 3020024310
]
asia_indices = [
    4020000010, 4020006940, 4020015090, 4020024190, 4020034510, 4020050210, 4020050220, 4020050290, 4020050470
]
australia_indices = [
    5020000010, 5020015660, 5020037270, 5020049720, 5020054880, 5020055870, 5020082270
]
south_america_indices = [
    6020000010, 6020006540, 6020008320, 6020014330, 6020017370, 6020021870, 6020029280
]
north_america_indices = [
    7020000010, 7020014250, 7020021430, 7020024600, 7020038340, 7020046750, 7020047840, 7020065090
]
arctic_na_indices = [
    8020000010, 8020008900, 8020010700, 8020020760, 8020022890, 8020032840, 8020044560
]
greenland_indices = [9020000010]
all_indices = (north_america_indices + africa_indices + south_america_indices + siberia_indices + asia_indices
               + australia_indices + arctic_na_indices + europe_indices + greenland_indices)


def make_basin_inputs(index_list: list=all_indices):
    urls = []
    save_paths = []
    for basin_index in index_list:
        urls.append(f'https://earth-info.nga.mil/php/download.php?file={basin_index}-basins-gpkg')
        save_paths.append(ppaths.tdx_basins/f'basin_{basin_index}.gpkg')
    return urls, save_paths


def make_stream_inputs(index_list: list=all_indices):
    urls = []
    save_paths = []
    for basin_index in index_list:
        urls.append(f'https://earth-info.nga.mil/php/download.php?file={basin_index}-streamnet-gpkg')
        save_paths.append(ppaths.tdx_streams / f'basin_{basin_index}.gpkg')
    return urls, save_paths


def check_complete(basin_index: int, base_dir: Path):
    progress_path = base_dir/f'basin_{basin_index}_progress.txt'
    if progress_path.exists():
        with open(progress_path, 'r') as src:
            text = src.read()
        if '100%' in text:
            print(f'{basin_index} is complete')
            return True
    return False


def convert_to_parquet(save_path: Path):
    parent_path = save_path.parent
    parquet_name = save_path.name.replace('.gpkg', '.parquet')
    parquet_path = parent_path/parquet_name
    if not parquet_path.exists():
        gdf = gpd.read_file(save_path)
        gdf.to_parquet(parquet_path)


if __name__ == '__main__':
    number_of_processes = 20

    basin_urls, basin_paths = make_basin_inputs()
    multi_download(basin_urls, basin_paths, num_proc=number_of_processes)

    stream_urls, stream_paths = make_stream_inputs()
    multi_download(stream_urls, stream_paths, num_proc=number_of_processes)

