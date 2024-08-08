import os
from water.basic_functions import ppaths, Path
import shutil
import rasterio as rio

def copy_tif_files(
        continent_dir: Path, output_dir: Path, overwrite: bool=False
) -> None:
    if not output_dir.exists():
        output_dir.mkdir()
    for country_dir in continent_dir.iterdir():
        output_data = country_dir/'output_data_merged'
        if output_data.exists():
            for file in output_data.iterdir():
                if not (output_dir/file.name).exists() or overwrite:
                    shutil.copy(src=file, dst=output_dir/file.name)
        else:
            copy_country_subdir_tif_files(country_dir, output_dir, overwrite)
    for file in output_dir.glob('*.tif'):
        with rio.open(file, 'r+') as rio_f:
            rio_f.nodata = 0


def copy_country_subdir_tif_files(country_dir: Path, output_dir: Path, overwrite: bool=False) -> None:
    country = country_dir.name
    for subdir in country_dir.iterdir():
        output_data = subdir/'output_data_merged'
        if output_data.exists():
            for file in output_data.iterdir():
                file_name = f'{country}_{file.name}'
                if not (output_dir/file_name).exists() or overwrite:
                    shutil.copy(src=file, dst=output_dir/file_name)


def copy_parquet_files(
        continent_dir: Path, output_dir: Path, overwrite: bool=False
) -> None:
    if not output_dir.exists():
        output_dir.mkdir()
    for country_dir in continent_dir.iterdir():
        for file in country_dir.glob('*_waterways_model.parquet'):
            if not (output_dir/file.name).exists() or overwrite:
                shutil.copy(src=file, dst=output_dir/file.name)



if __name__ == '__main__':
    # continent = 'oceania'
    # continent_dir = ppaths.country_data/continent
    # output_dir_raster = ppaths.country_data/f'all_countries_raster_files'
    # # output_dir_vector = ppaths.country_data/f'{continent}_vector_files'
    # copy_tif_files(continent_dir=continent_dir, output_dir=output_dir_raster)
    # # copy_parquet_files(continent_dir=continent_dir, output_dir=output_dir_vector)
    continent = 'africa'
    print(f'copying {continent}')
    country_dir = ppaths.country_data/f'{continent}'
    output_dir_raster = ppaths.country_data/f'all_countries_raster_files'
    copy_tif_files(continent_dir=country_dir, output_dir=output_dir_raster)

    continent = 'oceania'
    print(f'copying {continent}')
    country_dir = ppaths.country_data/f'{continent}'
    output_dir_raster = ppaths.country_data/f'all_countries_raster_files'
    copy_tif_files(continent_dir=country_dir, output_dir=output_dir_raster)
    # # copy_parquet_files(continent_dir=continent_dir, output_dir=output_dir_vector)

    # continent = 'argentina'
    # continent_dir = ppaths.country_data/f'america/{continent}'
    # copy_tif_files(continent_dir=continent_dir, output_dir=output_dir_raster)
    # copy_parquet_files(continent_dir=continent_dir, output_dir=output_dir_vector)
    #
    # continent = 'brazil'
    # continent_dir = ppaths.country_data/f'america/{continent}'
    # copy_tif_files(continent_dir=continent_dir, output_dir=output_dir_raster)
    # copy_parquet_files(continent_dir=continent_dir, output_dir=output_dir_vector)

    # continent = 'europe'
    # continent_dir = ppaths.country_data/continent
    # output_dir_raster = ppaths.country_data/f'{continent}_raster_files'
    # output_dir_vector = ppaths.country_data/f'{continent}_vector_files'
    # copy_tif_files(continent_dir=continent_dir, output_dir=output_dir_raster)
    # copy_parquet_files(continent_dir=continent_dir, output_dir=output_dir_vector)
    #
    # continent = 'africa'
    # continent_dir = ppaths.country_data/continent
    # output_dir_raster = ppaths.country_data/f'{continent}_raster_files'
    # output_dir_vector = ppaths.country_data/f'{continent}_vector_files'
    # copy_tif_files(continent_dir=continent_dir, output_dir=output_dir_raster)
    # copy_parquet_files(continent_dir=continent_dir, output_dir=output_dir_vector)