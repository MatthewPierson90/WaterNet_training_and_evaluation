import pandas as pd

from water.make_country_waterways.prepare_data_main import prepare_main
from water.make_country_waterways.evaluate_data_stream import evaluate_on_all_sen_data_multi
from water.make_country_waterways.add_weight import add_weight_to_all_outputs
from water.make_country_waterways.make_country_grids import make_grids
from water.make_country_waterways.cut_data import cut_data_to_match, merge_dir_and_save
from water.make_country_waterways.make_country_elevation_tif import make_country_elevation_tif
from water.basic_functions import (get_country_bounding_box, ppaths, Path,
                                   resuffix_directory_and_make_new, delete_directory_contents)
from water.make_country_waterways.vectorize_outputs import vectorize_and_save
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_country_evaluate(
        country: str, num_proc: int, eval_grid_width: int, model_number: int, num_per: int,
        output_grid_width: int = None, data_dir_names: list = ('elevation', 'sentinel_4326'),
        base_dir_path: Path = ppaths.waterway/'country_data', recut_output_data=False, output_grid_res=40,
        small_land_count=100, min_water_val=.3, output_dir_name='output_data', save_name='waterways_model.parquet',
        eval_grid_step_size: int = None, **kwargs
):
    country_path = base_dir_path/country
    if not country_path.exists():
        country_path.mkdir()
    output_grid_width = output_grid_width if output_grid_width is not None else eval_grid_width
    eval_grid_step_size = eval_grid_width if eval_grid_step_size is None else eval_grid_step_size
    print(f'Preparing temp files for {country}')
    prepare_main(
        country=country, grid_width=eval_grid_width, step_size=eval_grid_step_size, base_dir_path=base_dir_path,
        save_dir_name='temp', grid_res=10, bbox=get_country_bounding_box(country)
    )
    print('\n')
    print(f'Evaluating for {country}')
    evaluate_on_all_sen_data_multi(
        model_number=model_number, num_per=num_per, country=country, base_dir_path=base_dir_path,
        num_proc=num_proc, output_name=output_dir_name, **kwargs
    )
    wait(10)
    add_weight_to_all_outputs(
        country=country, base_dir_path=base_dir_path, num_proc=num_proc, output_name=output_dir_name
    )
    wait(10)
    to_merge_dir_name = output_dir_name
    if recut_output_data:
        print('\n')
        print(f'cutting output for {country}')
        make_grids(
                country=country, bbox=get_country_bounding_box(country), base_dir_path=base_dir_path,
                step_size=output_grid_width, grid_width=output_grid_width,
                grid_res=output_grid_res, save_dir_name=f'{output_dir_name}_temp',
        )
        wait(10)
        cut_data_to_match(
            country=country, num_proc=num_proc, match_dir_name=f'{output_dir_name}_temp',
            data_dir=country_path/f'{output_dir_name}_weighted', use_mean_merge=True,
            base_dir_path=base_dir_path
        )
        to_merge_dir_name = f'{output_dir_name}_weighted_cut'
    print('\n')
    wait(10)
    if not (country_path/f'{output_dir_name}_merged').exists():
        (country_path/f'{output_dir_name}_merged').mkdir()
    else:
        resuffix_directory_and_make_new(country_path/f'{output_dir_name}_merged')
    merge_dir_and_save(
        dir_path=country_path/to_merge_dir_name, save_dir_path=country_path/f'{output_dir_name}_merged',
        save_name=f'{country}_merged_data.tif'
    )
    delete_directory_contents(country_path/output_dir_name)
    delete_directory_contents(country_path/f'{output_dir_name}_weighted')
    wait(20)
    make_country_elevation_tif(
        country=country, output_dir=country_path/'output_data_merged',
        elevation_path=base_dir_path/'elevation', base_dir=base_dir_path
    )
    wait(20)
    print(f'vectorizing outputs for {country}')
    vectorize_and_save(
        country=country, country_dir=country_path,
        num_proc=1, output_dir_name=country_path/f'{output_dir_name}_merged',
        min_water_val=min_water_val, small_land_count=small_land_count, save_name=save_name
    )
    delete_directory_contents(country_path/'elevation_merged')
    delete_directory_contents(country_path/'output_data_weighted_cut')
    delete_directory_contents(country_path/'output_data_temp')
    delete_directory_contents(country_path/'sentinel_4326_cut')
    delete_directory_contents(country_path/'elevation_cut')
    delete_directory_contents(country_path/'input_data')
    delete_directory_contents(country_path/'vector_data')
    delete_directory_contents(country_path/'temp')


def wait(time: float):
    s = tt()
    while tt() - s < time:
        continue


if __name__ == '__main__':
    from water.basic_functions import tt, time_elapsed
    num_p = 20
    model_num = 662
    eval_scale = 2
    num_per_exp = 4 - eval_scale
    eval_grid_width = 832
    recut_output = True
    output_width = 416
    # small_land_count = 0
    output_grid_res = 40
    small_land_count = 25
    water_accept_per = .40
    num_per = 32
    output_dir_name ='output_data'
    save_name = 'waterways_model'
    print(num_per, eval_grid_width)
    # countries = ['rwanda', 'uganda']
    inputs = dict(
        eval_grid_width=eval_grid_width, eval_grid_step_size=eval_grid_width,
        model_number=model_num, small_land_count=small_land_count, min_water_val=water_accept_per,
        num_proc=num_p, num_per=num_per, recut_output_data=recut_output, output_grid_width=output_width,
        output_dir_name=output_dir_name, save_name=save_name, output_grid_res=output_grid_res
    )
    # countries = ['andorra', 'luxembourg', 'portugal']
    # for index, ctry in enumerate(countries):
    #     print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     print(f'\nWorking on {ctry} ({index+1}/{len(countries)})\n')
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    #
    #     # if not (ppaths.country_data/f'africa/{ctry}').exists():
    #     s = tt()
    #     inputs.update({'country': ctry, 'base_dir_path': ppaths.country_data/'europe'})
    #     p = Process(target=run_country_evaluate, kwargs=inputs)
    #     p.start()
    #     p.join()
    #     p.close()
    #     print('')
    #     time_elapsed(s)

    countries = [
        'paraguay', 'ecuador', 'guyana', 'uruguay', 'suriname', 'french_guiana'
    ]
    for index, ctry in enumerate(countries):
        print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f'\nWorking on {ctry} ({index+1}/{len(countries)})\n')
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

        # if not (ppaths.country_data/f'africa/{ctry}').exists():
        s = tt()
        inputs.update({'country': ctry, 'base_dir_path': ppaths.country_data/'america'})

        p = Process(target=run_country_evaluate, kwargs=inputs)
        p.start()
        p.join()
        p.close()
        print('')
        time_elapsed(s)
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


    # countries = ['rwanda', 'uganda', 'ivory_coast', 'kenya', 'zambia', 'ethiopia', 'tanzania']
    # for index, ctry in enumerate(countries):
    #     print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     print(f'\nWorking on {ctry} ({index+1}/{len(countries)})\n')
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    #
    #     # if not (ppaths.country_data/f'africa/{ctry}').exists():
    #     s = tt()
    #     inputs.update({'country': ctry, 'base_dir_path': ppaths.country_data/'africa'})
    #
    #     p = Process(target=run_country_evaluate, kwargs=inputs)
    #     p.start()
    #     p.join()
    #     p.close()
    #     print('')
    #     time_elapsed(s)
    #
    # seen_countries = set(countries)
    # # countries = ['rwanda']
    # def get_area(bbox_dict):
    #     w = bbox_dict['sw']['lon']
    #     s = bbox_dict['sw']['lat']
    #     e = bbox_dict['ne']['lon']
    #     n = bbox_dict['ne']['lat']
    #     return (n-s)*(e-w)
    # country_info = pd.read_parquet(ppaths.country_lookup_data/'country_information.parquet')
    # country_info['area'] = country_info.boundingBox.apply(get_area)
    # country_info = country_info.sort_values(by='area')
    # countries = country_info[country_info.continent == 'Africa'].country_name.to_list()
    # print(countries)
    # # for ind, ctry in enumerate(countries):
    # #     print(ind, ctry)
    # # s = tt()
    # # countries = ['democratic_republic_of_the_congo']
    # # countries = ['south_sudan']
    #
    # for index, ctry in enumerate(countries):
    #     if ctry not in seen_countries:
    #         print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #         print(f'\nWorking on {ctry} ({index+1}/{len(countries)})\n')
    #         print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    #
    #         # if not (ppaths.country_data/f'africa/{ctry}').exists():
    #         s = tt()
    #         inputs.update({'country': ctry, 'base_dir_path': ppaths.country_data/'africa'})
    #         p = Process(target=run_country_evaluate, kwargs=inputs)
    #         p.start()
    #         p.join()
    #         p.close()
    #         print('')
    #         time_elapsed(s)
    # #
