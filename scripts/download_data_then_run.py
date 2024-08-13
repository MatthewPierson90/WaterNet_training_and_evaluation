from water.basic_functions import ppaths, tt, time_elapsed, get_country_polygon, delete_directory_contents
from water.data_functions.download.merge_sentinel_tiles import merge_save_and_remove_multi
from water.data_functions.download.download_mpc_data import download_country_list_tiles, CloudSearchInfo
from water.data_functions.download.download_elevation import download_country_elevation_data
from water.deployment_functions.deploy_on_polygon import run_polygon_evaluate, get_admin_gdf_from_country_name
from multiprocessing import Process
from water.update_progress import make_progress_file, update_progress

import shapely
import os
progress_path = ppaths.base_path/'progress.txt'



# country_list = [
#     'Philippines', 'Malaysia', 'Singapore', 'Brunei', 'Indonesia',
#     'west bank'
# ]
# continent_tile_dir = ppaths.evaluation_data / 'sentinel_tiles_asia'
# download_inputs = dict(
#     num_proc=8, time_of_interest='2023-03-01/2023-9-30',
#     save_dir_path=continent_tile_dir,
#     cloud_info=CloudSearchInfo(1, 25, 5),
#     max_items=4, min_year=2023, min_intersection_percent=.3,
#     max_percent_remaining=.01
# )
# merge_inputs = dict(
#     base_dir=continent_tile_dir, save_dir=ppaths.evaluation_data/'sentinel_4326', num_proc=6,
#     num_steps=500
# )
# percent_remaining_list = [.6, .7, .8]
# for max_percent_remaining in percent_remaining_list:
#     progress_message = f'Downloading data for country_list, max_percent_remaining is {max_percent_remaining}.'
#     update_progress(progress_path, progress_message)
#     print(progress_message, '\n')
#     download_inputs.update({'max_percent_remaining': max_percent_remaining})
#     download_country_list_tiles(country_list, **download_inputs)
#
#     s = tt()
#     progress_message = f'Merging data for country_list, max_percent_remaining is {max_percent_remaining}.\n'
#     update_progress(progress_path, progress_message)
#     print(progress_message)
#     merge_save_and_remove_multi(**merge_inputs)
#     te = time_elapsed(s, 2)
#
#     if tt()-s < 10:
#         break
#
# for country in country_list:
#     download_country_elevation_data(country, num_proc=8)



# country_list = ['new guinea', 'new caledonia', 'new zealand']
continent_tile_dir = ppaths.evaluation_data / 'sentinel_tiles_russia'
download_inputs = dict(
    num_proc=8, time_of_interest='2023-04-01/2023-08-30',
    save_dir_path=continent_tile_dir,
    cloud_info=CloudSearchInfo(1, 25, 5),
    max_items=2, min_year=2023, min_intersection_percent=0,
    max_percent_remaining=.01
)
merge_inputs = dict(
    base_dir=continent_tile_dir, save_dir=ppaths.evaluation_data/'sentinel_4326', num_proc=10,
    num_steps=500
)

download_inputs.update({'max_percent_remaining': .01})
country_list = ['russia']
for country in country_list:
    percent_remaining_list = [.99]
    for max_percent_remaining in percent_remaining_list:
        progress_message = f'Merging data for {country}, max_percent_remaining is {max_percent_remaining}.\n'
        update_progress(progress_path, progress_message)
        merge_save_and_remove_multi(**merge_inputs)

        progress_message = f'Downloading data for {country}, max_percent_remaining is {max_percent_remaining}.'
        update_progress(progress_path, progress_message)
        download_inputs.update({'max_percent_remaining': max_percent_remaining})
        download_country_list_tiles([country], **download_inputs)
    # download_country_elevation_data(country, num_proc=8)





# num_p = 20
# model_num = 662
# eval_scale = 2
# num_per_exp = 4 - eval_scale
# eval_grid_width = 832
# recut_output = True
# output_width = 416
# output_grid_res = 40
# small_land_count = 25
# water_accept_per = .40
# num_per = 32
# output_dir_name ='output_data'
# save_name = 'waterways_model'
# print(num_per, eval_grid_width)
# inputs = dict(
#     eval_grid_width=eval_grid_width, eval_grid_step_size=eval_grid_width,
#     model_number=model_num, small_land_count=small_land_count, min_water_val=water_accept_per,
#     num_proc=num_p, num_per=num_per, recut_output_data=recut_output, output_grid_width=output_width,
#     output_dir_name=output_dir_name, save_name=save_name, output_grid_res=output_grid_res,
#     base_dir_path=ppaths.evaluation_data
# )



# country_list = ['china']
#
# continent_tile_dir = ppaths.evaluation_data / 'sentinel_tiles_asia'
# download_inputs = dict(
#     num_proc=8, time_of_interest='2023-03-01/2023-9-30',
#     save_dir_path=continent_tile_dir,
#     cloud_info=CloudSearchInfo(1, 25, 5),
#     max_items=4, min_year=2023, min_intersection_percent=.3,
#     max_percent_remaining=.01
# )
# merge_inputs = dict(
#     base_dir=continent_tile_dir, save_dir=ppaths.evaluation_data/'sentinel_4326', num_proc=6,
#     num_steps=500
# )
# for country in country_list:
#     percent_remaining_list = [.01, .01, .02, .05, .1, .2, .4, .6, .7]
#     for max_percent_remaining in percent_remaining_list:
#         progress_message = f'Merging data for asia, max_percent_remaining is {max_percent_remaining}.\n'
#         update_progress(progress_path, progress_message)
#         merge_save_and_remove_multi(**merge_inputs)
#
#         progress_message = f'Downloading data for asia, max_percent_remaining is {max_percent_remaining}.'
#         update_progress(progress_path, progress_message)
#         download_inputs.update({'max_percent_remaining': max_percent_remaining})
#         download_country_list_tiles([country], **download_inputs)
#     download_country_elevation_data(country, num_proc=8)
#
#
#
# country_list = ['russia']
#
# continent_tile_dir = ppaths.evaluation_data / 'sentinel_tiles_asia'
# download_inputs = dict(
#     num_proc=8, time_of_interest='2023-03-01/2023-9-30',
#     save_dir_path=continent_tile_dir,
#     cloud_info=CloudSearchInfo(1, 25, 5),
#     max_items=4, min_year=2023, min_intersection_percent=.3,
#     max_percent_remaining=.01
# )
# merge_inputs = dict(
#     base_dir=continent_tile_dir, save_dir=ppaths.evaluation_data/'sentinel_4326', num_proc=6,
#     num_steps=500
# )
# for country in country_list:
#     percent_remaining_list = [.01, .01, .02, .05, .1, .2, .4, .6, .7]
#     for max_percent_remaining in percent_remaining_list:
#         progress_message = f'Merging data for asia, max_percent_remaining is {max_percent_remaining}.\n'
#         update_progress(progress_path, progress_message)
#         merge_save_and_remove_multi(**merge_inputs)
#
#         progress_message = f'Downloading data for asia, max_percent_remaining is {max_percent_remaining}.'
#         update_progress(progress_path, progress_message)
#         download_inputs.update({'max_percent_remaining': max_percent_remaining})
#         download_country_list_tiles([country], **download_inputs)
#     download_country_elevation_data(country, num_proc=8)

# country_list = [
#     'syria', 'iraq', 'iran', 'lebanon', 'jordan', 'israel', 'saudi_arabia', 'oman', 'yemen', 'kuwait',
#     'bahrain', 'qatar', 'united_arab_emirates', 'armenia', 'georgia', 'azerbaijan', 'turkmenistan',
#     'uzbekistan', 'afghanistan', 'pakistan', 'tajikistan', 'kyrgyzstan', 'kazakhstan'
# ]
# progress_message = f'Downloading elevation data.'
# update_progress(progress_path, progress_message)
# country_list = [
#     'iran'
# ]
# for country in country_list:
#     print(country)
#     download_country_elevation_data(country, num_proc=8)
#
# os.system('mullvad disconnect')

# num_p = 20
# model_num = 662
# eval_scale = 2
# num_per_exp = 4 - eval_scale
# eval_grid_width = 832
# recut_output = True
# output_width = 416
# # small_land_count = 0
# output_grid_res = 40
# small_land_count = 25
# water_accept_per = .40
# num_per = 32
# output_dir_name ='output_data'
# save_name = 'waterways_model'
# print(num_per, eval_grid_width)
# # countries = ['rwanda', 'uganda']
# inputs = dict(
#     eval_grid_width=eval_grid_width, eval_grid_step_size=eval_grid_width,
#     model_number=model_num, small_land_count=small_land_count, min_water_val=water_accept_per,
#     num_proc=num_p, num_per=num_per, recut_output_data=recut_output, output_grid_width=output_width,
#     output_dir_name=output_dir_name, save_name=save_name, output_grid_res=output_grid_res,
#     base_dir_path=ppaths.evaluation_data
# )
# countries = [
#     'syria', 'iraq', 'iran', 'lebanon', 'jordan', 'israel', 'saudi_arabia', 'oman', 'yemen', 'kuwait',
#     'bahrain', 'qatar', 'united_arab_emirates', 'armenia', 'georgia', 'azerbaijan', 'turkmenistan',
#     'uzbekistan', 'afghanistan', 'pakistan', 'tajikistan', 'kyrgyzstan', 'kazakhstan'
#     ]
# countries = [
#     'syria', 'iraq', 'lebanon', 'jordan', 'israel', 'saudi_arabia', 'oman', 'yemen', 'kuwait',
#     'bahrain', 'qatar', 'united_arab_emirates', 'georgia', 'turkmenistan',
#     'uzbekistan', 'afghanistan', 'pakistan', 'tajikistan', 'kyrgyzstan', 'kazakhstan'
#     ]
# countries = [
#     'georgia'
#     ]
# country_dir = ppaths.evaluation_data/'asia'
# if not country_dir.exists():
#     country_dir.mkdir()
# for ctry_index, ctry in enumerate(countries):
#     progress_message = f'Evaluating {ctry} ({ctry_index+1}/{len(countries)}).'
#     update_progress(progress_path, progress_message)
#     print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(f'\nWorking on {ctry} ({ctry_index+1}/{len(countries)})\n')
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
#     try:
#         polygon = get_country_polygon(country=ctry)
#         name = ctry.lower().replace(' ', '_').replace('-', '_').replace('\'', '_')
#         save_dir = country_dir/name
#         if not save_dir.exists():
#             s = tt()
#             if not save_dir.exists():
#                 save_dir.mkdir(parents=True)
#             inputs.update(
#                 {'polygon': polygon, 'polygon_name': name, 'save_dir_path': save_dir}
#             )
#             p = Process(target=run_polygon_evaluate, kwargs=inputs)
#             p.start()
#             p.join()
#             p.close()
#             print('')
#             time_elapsed(s)
#     except:
#         continue



