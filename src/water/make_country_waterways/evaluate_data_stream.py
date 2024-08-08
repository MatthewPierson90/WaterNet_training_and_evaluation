import torch
# from water.training.model_container import ModelContainer, load_model_container
from water.training.model_container import ModelTrainingContainer
from water.make_country_waterways.merge_prepared_data import open_all_and_merge
from water.make_country_waterways.cut_data import cut_data_to_match_file_list

import numpy as np
import rasterio as rio
from water.basic_functions import ppaths, tt, time_elapsed, delete_directory_contents, save_pickle, open_pickle
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt

from multiprocessing import Process


def fill_missing_data(data):
    if len(data[np.isnan(data)]) < .5*len(data.flatten()):
    # if True:
        for ch in range(data.shape[0]):
            if np.any(np.isnan(data[ch])):
                nan_mean = np.nanmean(data[ch], dtype=np.float32)
                nan_mean = nan_mean if not np.isnan(nan_mean) else 0
                data[ch] = np.nan_to_num(data[ch], nan=nan_mean).astype(np.float32)
        # ch, r, c = np.where(np.isnan(data))
        # data[ch, r, c] = np.nanmean(data[ch].astype(np.float32), dtype=np.float32)
    # else:
    #     print(len(data[np.isnan(data)])/len(data.flatten()))
    #     for ch in range(data.shape[0]):
    #         print('  ', len(data[ch][np.isnan(data[ch])])/len(data[ch].flatten()))
    return data


def save_model_outputs(model_output, file_paths, save_dir):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    for file_ind, file_path in enumerate(file_paths):
        save_path = save_dir/file_path.name
        with rio.open(file_path) as rio_f:
            save_profile = rio_f.profile
            _, count, height, width = model_output.shape
            w, s, e, n = rio_f.bounds
            transform = from_bounds(
                west=w, south=s, east=e, north=n, width=width, height=height
            )
            save_profile['transform'] = transform
            save_profile['height'] = height
            save_profile['width'] = width
        save_profile['count'] = count
        save_profile['dtype'] = 'float32'
        save_profile['nodata'] = np.nan
        with rio.open(save_path, 'w', **save_profile) as dst_f:
            dst_f.write(model_output[file_ind])


def evaluate_model_on_single_image(model_container: ModelTrainingContainer, data):
    output = model_container.evaluate_model(model_name='wwm', train=False, input=data)
    output_cpu = output.detach().to('cpu').float()
    del data
    del output
    return output_cpu


def cut_next_file_set(
        file_paths, country, use_veg_indices, num_proc, country_dir, base_dir_path, eval_grid_width=832
):
    delete_directory_contents(country_dir/'elevation_cut')
    delete_directory_contents(country_dir/'sentinel_4326_cut')
    cut_data_to_match_file_list(
        file_paths=file_paths.copy(), save_dir=country_dir/'elevation_cut',
        data_dir=base_dir_path/'elevation', num_proc=num_proc, use_mean_merge=False
    )
    cut_data_to_match_file_list(
        file_paths=file_paths.copy(), save_dir=country_dir/'sentinel_4326_cut',
        data_dir=base_dir_path/'sentinel_4326', num_proc=num_proc, use_mean_merge=False
    )
    data_list = []
    file_list = []
    for ind, file in enumerate(file_paths):
        sentinel_path = country_dir/f'sentinel_4326_cut/{file.name}'
        # print(file)
        elevation_path = country_dir/f'elevation_cut/{sentinel_path.name}'

        data, _, _ = open_all_and_merge(
            sentinel_path=sentinel_path, elevation_path=elevation_path, make_indices=use_veg_indices,
            shape=(1, eval_grid_width, eval_grid_width)
        )
        data = fill_missing_data(data)
        if not np.any(np.isnan(data)):
            data = np.stack([data], axis=0)
            file_list.append(file)
            data_list.append(data)
    if len(file_list) > 0:
        data = np.concatenate(data_list, axis=0)
    else:
        data = np.array([])
    save_pickle(country_dir/f'input_data/next_inputs_files.pkl', file_list)
    # print(len(file_list))
    np.save(country_dir/f'input_data/next_inputs.npy', data)


def evaluate_on_all_sen_data_multi(model_number,
                                   country,
                                   use_veg_indices=True,
                                   num_per=1,
                                   save_inputs=False,
                                   output_name='output_data',
                                   base_dir_path=ppaths.country_data,
                                   country_dir=None,
                                   num_proc=16,
                                   eval_grid_width=832,
                                   **kwargs
                                   ):
    if country_dir is None:
        country_dir = base_dir_path/country
    with torch.no_grad():
        model_container = ModelTrainingContainer.load_container(model_number=model_number)
        inputs_dir = country_dir/'input_data'
        outputs_dir = country_dir/output_name
        sen_dir = country_dir/'sentinel_4326_cut'
        el_dir = country_dir/'elevation_cut'
        for dir in [inputs_dir, outputs_dir, sen_dir, el_dir]:
            if not dir.exists():
                dir.mkdir()
            else:
                delete_directory_contents(dir)
        temp_paths = list((country_dir/'temp').glob('*'))
        num_files = len(temp_paths)
        current_paths = temp_paths[0:num_per]
        cut_next_file_set(
            file_paths=temp_paths[0:num_per], country=country, country_dir=country_dir,
            use_veg_indices=use_veg_indices, num_proc=num_proc, base_dir_path=base_dir_path,
            eval_grid_width=eval_grid_width
        )
        file_ind = num_per
        current_ind = 0
        s = tt()
        while file_ind <= num_files - 1:
            file_paths = temp_paths[file_ind:file_ind+num_per]
            inputs = dict(
                file_paths=file_paths, country=country, country_dir=country_dir,
                use_veg_indices=use_veg_indices, num_proc=num_proc, base_dir_path=base_dir_path,
                eval_grid_width=eval_grid_width
            )
            proc = Process(target=cut_next_file_set, kwargs=inputs)
            proc.start()
            data = torch.tensor(
                np.load(inputs_dir/'next_inputs.npy'), dtype=model_container.dtype, device=model_container.device
            )
            output = torch.tensor([])
            current_paths = open_pickle(country_dir/f'input_data/next_inputs_files.pkl')
            if len(current_paths) > 0:
                output = evaluate_model_on_single_image(model_container, data)
                save_model_outputs(output, current_paths, outputs_dir)
            # current_paths = file_paths
            proc.join()
            proc.close()
            file_ind += num_per
            if file_ind / (num_files // 20) > current_ind and file_ind > max(num_files // 40, 1):
                current_ind += 1
                print(f'Completed {(file_ind + 1) / num_files:.2%} ({file_ind + 1}/{num_files})')
                time_elapsed(s, 2)
            del data
            del output
        data = torch.tensor(
            np.load(inputs_dir/'next_inputs.npy'), dtype=model_container.dtype, device=model_container.device
        )
        current_paths = open_pickle(country_dir/f'input_data/next_inputs_files.pkl')
        if len(current_paths)>0:
            output = evaluate_model_on_single_image(model_container, data)
            save_model_outputs(output, current_paths, outputs_dir)
    for model in model_container.model_container.model_dict.values():
        model.to('cpu')
    del model_container
    torch.cuda.empty_cache()


