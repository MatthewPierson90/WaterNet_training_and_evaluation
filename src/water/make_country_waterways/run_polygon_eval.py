import pandas as pd
import shapely

from water.make_country_waterways.prepare_data_main import prepare_main
from water.make_country_waterways.evaluate_data_stream import evaluate_on_all_sen_data_multi
from water.make_country_waterways.add_weight import add_weight_to_all_outputs
from water.make_country_waterways.make_country_grids import make_grids
from water.make_country_waterways.cut_data import cut_data_to_match, merge_dir_and_save
from water.make_country_waterways.make_country_elevation_tif import make_country_elevation_tif
from water.basic_functions import (get_country_bounding_box, ppaths, Path, get_alpha_3_code_from_country_name,
                                   resuffix_directory_and_make_new, delete_directory_contents)
from water.make_country_waterways.vectorize_outputs import vectorize_and_save
from multiprocessing import Process
from water.basic_functions import tt, time_elapsed, get_country_polygon, get_alpha_3_code_from_country_name

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_polygon_evaluate(
        polygon: shapely.Polygon or shapely.MultiPolygon,
        polygon_name: str,
        save_dir_path: Path,
        num_proc: int, eval_grid_width: int, model_number: int, num_per: int,
        output_grid_width: int = None, data_dir_names: list = ('elevation', 'sentinel_4326'),
        base_dir_path: Path = ppaths.waterway/'country_data', recut_output_data=False, output_grid_res=40,
        small_land_count=100, min_water_val=.3, output_dir_name='output_data', save_name='waterways_model.parquet',
        eval_grid_step_size: int = None, **kwargs
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
    evaluate_on_all_sen_data_multi(
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
        # cut_data_to_match(
        #     match_dir=save_dir_path/f'{output_dir_name}_temp',
        #     save_dir=save_dir_path/f'{output_dir_name}_cut',
        #     data_dir=save_dir_path/f'{output_dir_name}',
        #     use_mean_merge=True, num_proc=num_proc
        # )
        # to_merge_dir_name = f'{output_dir_name}_cut'
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
    # delete_directory_contents(save_dir_path/output_dir_name)
    # delete_directory_contents(save_dir_path/f'{output_dir_name}_weighted')

    # wait(10)
    # make_country_elevation_tif(
    #     save_dir_path=save_dir_path, output_dir=save_dir_path/'output_data_merged',
    #     elevation_path=base_dir_path/'elevation'
    # )
    # wait(10)
    # print(f'vectorizing outputs for {polygon_name}')
    # vectorize_and_save(
    #     country=polygon_name, country_dir=save_dir_path,
    #     num_proc=1, output_dir_name=save_dir_path/f'{output_dir_name}_merged',
    #     min_water_val=min_water_val, small_land_count=small_land_count, save_name=save_name
    # )
    # delete_directory_contents(save_dir_path/'elevation_merged')
    # delete_directory_contents(save_dir_path/'output_data_cut')
    # # delete_directory_contents(save_dir_path/'output_data_weighted_cut')
    # delete_directory_contents(save_dir_path/'output_data_temp')
    # delete_directory_contents(save_dir_path/'sentinel_4326_cut')
    # delete_directory_contents(save_dir_path/'elevation_cut')
    # delete_directory_contents(save_dir_path/'input_data')
    # delete_directory_contents(save_dir_path/'vector_data')
    # delete_directory_contents(save_dir_path/'temp')


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


if __name__ == '__main__':
    import geopandas as gpd
    num_p = 24
    # model_num = 662 <- used to generate world raster dataset
    model_num = 825
    eval_scale = 2
    num_per_exp = 4 - eval_scale
    eval_grid_width = 832
    recut_output = True
    output_width = 832
    # output_width = 416
    # small_land_count = 0
    # output_grid_res = 40
    output_grid_res = 20
    small_land_count = 25
    water_accept_per = .40
    num_per = 32
    output_dir_name ='output_data'
    save_name = 'waterways_model'
    print(num_per, eval_grid_width)
    # countries = ['rwanda', 'uganda']
    inputs = dict(
        eval_grid_width=eval_grid_width, eval_grid_step_size=eval_grid_width//2,
        model_number=model_num, small_land_count=small_land_count, min_water_val=water_accept_per,
        num_proc=num_p, num_per=num_per, recut_output_data=recut_output, output_grid_width=output_width,
        output_dir_name=output_dir_name, save_name=save_name, output_grid_res=output_grid_res,
        base_dir_path=ppaths.country_data
    )

    def run_process(country_dir, name, polygon, index, num_indices):
        save_dir = country_dir/name
        if not save_dir.exists():
            print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f'\n  Working on {name} ({index + 1}/{num_indices})\n')
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
            s = tt()
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            inputs.update(
                {'polygon': polygon, 'polygon_name': name, 'save_dir_path': save_dir}
            )
            p = Process(target=run_polygon_evaluate, kwargs=inputs)
            p.start()
            p.join()
            p.close()
            print('')
            time_elapsed(s)

    # countries = ['russia']
    # for ctry_index, ctry in enumerate(countries):
    #     print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     print(f'\nWorking on {ctry} ({ctry_index+1}/{len(countries)})\n')
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    #     admin_1 = get_admin_gdf_from_country_name(country_name=ctry, admin_level=1)
    #     country_dir = ppaths.country_data/f'asia/{ctry}'
    #     for admin_index, (polygon, name) in enumerate(zip(admin_1.geometry, admin_1.name)):
    #         name = name.lower().replace(' ', '_')
    #         name = f'{ctry}_{name}'
    #         if polygon.intersects(shapely.box(-180, -90, 0, 90)):
    #             polygon1 = polygon.intersection(shapely.box(-180, -90, 0, 90))
    #             run_process(
    #                 country_dir=country_dir, name=f'{name}_0', polygon=polygon1,
    #                 index=admin_index, num_indices=len(admin_1)
    #             )
    #             polygon2 = polygon.intersection(shapely.box(0, -90, 180, 90))
    #             run_process(
    #                 country_dir=country_dir, name=f'{name}_1', polygon=polygon2,
    #                 index=admin_index, num_indices=len(admin_1)
    #             )
    #         else:
    #             run_process(country_dir, name, polygon, index=admin_index, num_indices=len(admin_1))


    countries = ['rwanda']
    # countries = ['ivory_coast', 'ethiopia', 'zambia', 'tanzania', 'uganda', 'kenya']

    # bboxes = [(shapely.box(-180, -90, 0, 90)), shapely.box(0, -90, 180, 90)]
    for ctry_index, ctry in enumerate(countries):
        print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f'\nWorking on {ctry} ({ctry_index+1}/{len(countries)})\n')
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
        polygon = get_country_polygon(country=ctry).buffer(.15)
        country_dir = ppaths.country_data/f'africa/'
        name = ctry.lower().replace(' ', '_')
        save_dir = country_dir/name
        # if not save_dir.exists():
        s = tt()
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        inputs.update(
            {'polygon': polygon, 'polygon_name': name, 'save_dir_path': save_dir}
        )
        p = Process(target=run_polygon_evaluate, kwargs=inputs)
        p.start()
        p.join()
        p.close()
        print('')
        time_elapsed(s)

    # countries = ['new_zealand']
    # polygon = get_country_polygon(country='new_zealand')
    # # gpd.GeoSeries([polygon]).plot()
    # bboxes = [(shapely.box(-180, -90, 0, 90)), shapely.box(0, -90, 180, 90)]
    # polygons_names = [(polygon.intersection(bbox), f'new_zealand_{ind}') for ind, bbox in enumerate(bboxes)]
    # # gpd.GeoSeries([p[0] for p in polygon_names]).plot()
    # for ctry_index, ctry in enumerate(countries):
    #     progress_message = f'Evaluating {ctry}.'
    #     print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     print(f'\nWorking on {ctry} ({ctry_index + 1}/{len(countries)})\n')
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    #     country_dir = ppaths.country_data/f'oceania/{ctry}'
    #     if not country_dir.exists():
    #         country_dir.mkdir()
    #     for admin_index, (polygon, name) in enumerate(polygons_names):
    #         name = name.lower().replace(' ', '_')
    #         save_dir = country_dir/name
    #         print(save_dir)
    #         # if not save_dir.exists():
    #         print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #         print(f'\n  Working on {name} ({admin_index + 1}/{len(polygons_names)})\n')
    #         print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    #         s = tt()
    #         # save_dir.mkdir(parents=True)
    #         inputs.update(
    #             {'polygon': polygon, 'polygon_name': name, 'save_dir_path': save_dir}
    #         )
    #         p = Process(target=run_polygon_evaluate, kwargs=inputs)
    #         p.start()
    #         p.join()
    #         p.close()
    #         print('')
    #         time_elapsed(s)



    # countries = [
    #     'paraguay', 'ecuador', 'guyana', 'uruguay', 'suriname', 'french_guiana'
    # ]
    # for index, ctry in enumerate(countries):
    #     print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     print(f'\nWorking on {ctry} ({index+1}/{len(countries)})\n')
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    #
    #     # if not (ppaths.country_data/f'africa/{ctry}').exists():
    #     s = tt()
    #     inputs.update({'country': ctry, 'base_dir_path': ppaths.country_data/'america'})
    #
    #     p = Process(target=run_country_evaluate, kwargs=inputs)
    #     p.start()
    #     p.join()
    #     p.close()
    #     print('')
    #     time_elapsed(s)
    # countries = [
    #     'chile', 'paraguay', 'ecuador', 'guyana', 'uruguay', 'suriname', 'french_guiana', 'argentina', 'brazil'
    # ]
    # for index, ctry in enumerate(countries):
    #     print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     print(f'\nWorking on {ctry} ({index+1}/{len(countries)})\n')
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    #
    #     # if not (ppaths.country_data/f'africa/{ctry}').exists():
    #     s = tt()
    #     inputs.update({'country': ctry, 'base_dir_path': ppaths.country_data/'america'})
    #
    #     p = Process(target=run_country_evaluate, kwargs=inputs)
    #     p.start()
    #     p.join()
    #     p.close()
    #     print('')
    #     time_elapsed(s)

