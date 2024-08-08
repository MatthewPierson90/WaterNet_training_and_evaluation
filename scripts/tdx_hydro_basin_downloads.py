import os
from water.basic_functions import ppaths, Path
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor
# basin_indices = [
#     7020000010, 7020014250, 7020021430, 7020024600, 7020038340, 7020046750,
#     7020047840, 7020065090, 2020003440, 2020000010, 2020018240, 2020024230,
#     2020033490, 2020041390, 2020057170, 2020065840, 2020071190,
# ]

africa_indices = [
    1020000010, 1020011530, 1020018110, 1020021940,
    1020027430, 1020034170, 1020035180, 1020040190
]

europe_indices = [
    2020000010, 2020003440, 2020018240, 2020024230,
    2020033490, 2020041390, 2020057170, 2020065840,
    2020071190
]

siberia_indices = [
    3020000010, 3020003790, 3020005240, 3020008670,
    3020009320, 3020024310
]

asia_indices = [
    4020000010, 4020006940, 4020015090, 4020024190,
    4020034510, 4020050210, 4020050220, 4020050290,
    4020050470
]

australia_indices = [
    5020000010, 5020015660, 5020037270, 5020049720,
    5020054880, 5020055870, 5020082270
]

south_america_indices = [
    6020000010, 6020006540, 6020008320, 6020014330,
    6020017370, 6020021870, 6020029280
]

north_america_indices = [
    7020000010, 7020014250, 7020021430, 7020024600,
    7020038340, 7020046750, 7020047840, 7020065090
]

arctic_na_indices = [
    8020000010, 8020008900, 8020010700, 8020020760,
    8020022890, 8020032840, 8020044560
]

greenland_indices = [9020000010]


def download_basins(basin_index):
    save_base_path = ppaths.waterway/'tdx_basins'
    url = f'https://earth-info.nga.mil/php/download.php?file={basin_index}-basins-gpkg'
    save_base_path.mkdir(exist_ok=True)
    save_path = ppaths.waterway/f'tdx_basins/basin_{basin_index}.gpkg'
    progress_path = ppaths.waterway/f'tdx_basins/basin_{basin_index}_progress.txt'
    save_parquet = ppaths.waterway/f'tdx_basins/basin_{basin_index}.parquet'
    if not save_parquet.exists():
        os.system(f'wget -c {url} -O {save_path} -o {progress_path}')
    # try:
    #     gdf = gpd.read_file(save_path)
    #     gdf.to_parquet(save_parquet)
    # except:
    #     pass

def download_streams(basin_index):
    save_base_path = ppaths.waterway/'tdx_streams'
    url = f'https://earth-info.nga.mil/php/download.php?file={basin_index}-streamnet-gpkg'
    save_base_path.mkdir(exist_ok=True)
    save_path = save_base_path/f'basin_{basin_index}.gpkg'
    progress_path = save_base_path/f'basin_{basin_index}_progress.txt'
    save_parquet = save_base_path/f'basin_{basin_index}.parquet'
    if not save_parquet.exists():
        print(f'working on {save_path.name}')
        os.system(f'wget {url} -O {save_path} -o {progress_path}')
        print(f'Completed {save_path.name}')
    # try:
    #     gdf = gpd.read_file(save_path)
    #     gdf.to_parquet(save_parquet)
    # except:
    #     pass


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
    # thread_pool = ThreadPoolExecutor(max_workers=20)\
    from multiprocessing import Process
    # all_indices = (north_america_indices
    #                + africa_indices
    #                + south_america_indices
    #                + siberia_indices
    #                + asia_indices
    #                + australia_indices
    #                + arctic_na_indices
    #                + europe_indices
    #                + greenland_indices)
    save_base_path = ppaths.country_data/'tdx_streams'
    file_list = list(save_base_path.glob('*.gpkg'))
    num_files = len(file_list)
    for ind, file in enumerate(file_list):
        print(f'Working on {file.name} ({ind+1}/{num_files})')
        p = Process(target=convert_to_parquet, args=([file]))
        p.start()
        p.join()
        p.close()

    # save_base_path = ppaths.waterway/'tdx_basins'
    # for file in save_base_path.iterdir():
    #     if 'parquet' in file.name:
    #         gpkg_name = file.name.split('.')[0] + '.gpkg'
    #         gpkg_path = save_base_path / gpkg_name
    #         if not gpkg_path.exists():
    #             gdf = gpd.read_parquet(file)
    #             gdf.to_file(gpkg_path, driver='GPKG')
    # for basin_index in all_indices:
    #     # check_complete(basin_index, base_dir=save_base_path)
    #     if check_complete(basin_index, base_dir=save_base_path):
    #         save_path = save_base_path/f'basin_{basin_index}.gpkg'
    #         save_parquet = save_base_path/f'basin_{basin_index}.parquet'
    #         progress_path = save_base_path/f'basin_{basin_index}_progress.txt'
    #         if not save_parquet.exists():
    #             gdf = gpd.read_file(save_path)
    #             gdf.to_parquet(save_parquet)
    #         os.remove(save_path)
    #         os.remove(progress_path)
    # thread_pool.map(download_basins, north_america_indices)
    # thread_pool.map(download_basins, all_indices)


# 'https://earth-info.nga.mil/php/download.php?file=7020000010-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=7020014250-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=7020021430-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=7020024600-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=7020038340-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=7020046750-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=7020047840-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=7020065090-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=2020003440-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=2020000010-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=2020018240-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=2020024230-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=2020033490-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=2020041390-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=2020057170-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=2020065840-basins-gpkg'
# 'https://earth-info.nga.mil/php/download.php?file=2020071190-basins-gpkg'