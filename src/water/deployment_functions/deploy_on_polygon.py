import pandas as pd
import geopandas as gpd
import shapely

from water.deployment_functions.prepare_data_main import prepare_main
from water.deployment_functions.predict_data_stream import predict_on_all_sen_data_multi
from water.deployment_functions.add_weight import add_weight_to_all_outputs
from water.deployment_functions.reference_grids import make_grids
from water.deployment_functions.cut_data import cut_data_to_match, merge_dir_and_save
from water.basic_functions import resuffix_directory_and_make_new
from water.paths import ppaths, Path
from water.basic_functions import tt, time_elapsed, get_country_polygon, get_alpha_3_code_from_country_name

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_polygon_evaluate(
        polygon: shapely.Polygon or shapely.MultiPolygon,
        polygon_name: str, save_dir_path: Path,
        num_proc: int, eval_grid_width: int, model_number: int, num_per: int,
        output_grid_width: int = None, data_dir_names: list = ('elevation', 'sentinel_4326'),
        base_dir_path: Path = ppaths.training_data/'country_data', recut_output_data=False, output_grid_res=40,
        output_dir_name='output_data', eval_grid_step_size: int = None, **kwargs
):
    if not save_dir_path.exists():
        save_dir_path.mkdir()
    output_grid_width = output_grid_width if output_grid_width is not None else eval_grid_width
    eval_grid_step_size = eval_grid_width if eval_grid_step_size is None else eval_grid_step_size
    print(f'Preparing temp files for {polygon_name}')
    prepare_main(
        save_dir=save_dir_path/'temp', polygon=polygon, grid_width=eval_grid_width, step_size=eval_grid_step_size,
        save_dir_name='temp', grid_res=10
    )
    print('\n')
    print(f'Evaluating for {polygon_name}')
    predict_on_all_sen_data_multi(
        model_number=model_number, num_per=num_per, country=polygon_name, base_dir_path=base_dir_path,
        country_dir=save_dir_path, num_proc=num_proc, output_name=output_dir_name, eval_grid_width=eval_grid_width,
        **kwargs
    )
    wait(2)
    add_weight_to_all_outputs(
        output_parent_dir=save_dir_path, num_proc=num_proc, output_name=output_dir_name
    )
    wait(2)
    to_merge_dir_name = output_dir_name
    if recut_output_data:
        print('\n')
        print(f'cutting output for {polygon_name}')
        make_grids(
                save_dir=save_dir_path/f'{output_dir_name}_temp', polygon=polygon,
                step_size=output_grid_width, grid_width=output_grid_width, grid_res=output_grid_res,
        )
        wait(2)
        cut_data_to_match(
            match_dir=save_dir_path/f'{output_dir_name}_temp',
            save_dir=save_dir_path/f'{output_dir_name}_weighted_cut',
            data_dir=save_dir_path/f'{output_dir_name}_weighted',
            use_mean_merge=True, num_proc=num_proc
        )
        to_merge_dir_name = f'{output_dir_name}_weighted_cut'
    print('\n')
    wait(2)
    if not (save_dir_path/f'{output_dir_name}_merged').exists():
        (save_dir_path/f'{output_dir_name}_merged').mkdir()
    else:
        resuffix_directory_and_make_new(save_dir_path/f'{output_dir_name}_merged')
    merge_dir_and_save(
        dir_path=save_dir_path/to_merge_dir_name, save_dir_path=save_dir_path/f'{output_dir_name}_merged',
        save_name=f'{polygon_name}_merged_data.tif'
    )


def wait(time: float):
    s = tt()
    while tt() - s < time:
        continue


def get_admin_gdf_from_country_name(country_name: str, admin_level: int):
    shapefile_path = ppaths.country_lookup_data/'shapefiles'
    alpha_3 = get_alpha_3_code_from_country_name(country_name)
    dir_path = shapefile_path / f'gadm41_{alpha_3}_shp'
    file_path = dir_path / f'gadm41_{alpha_3}_{admin_level}.shp'
    df = gpd.read_file(file_path)
    df['name'] = df[f'NAME_{admin_level}']
    return df


def get_largest_polygon(multi_polygon: shapely.MultiPolygon):
    largest_area = 0
    largest_polygon = None
    for polygon in multi_polygon.geoms:
        if polygon.area > largest_area:
            largest_area = polygon.area
            largest_polygon = polygon
    return largest_polygon
