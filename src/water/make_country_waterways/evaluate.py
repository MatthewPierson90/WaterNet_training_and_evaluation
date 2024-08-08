import torch
# from water.training.model_container import ModelContainer, load_model_container
from water.training.model_containerV2 import ModelTrainingContainer
from water.make_country_waterways.merge_prepared_data import open_all_and_merge
from water.make_country_waterways.cut_data import cut_data_to_match_file_list
import numpy as np
import rasterio as rio
from water.basic_functions import ppaths, tt, time_elapsed, delete_directory_contents
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt



def fill_missing_data(data):
    if np.any(np.isnan(data)):
        for ch in range(data.shape[0]):
            if np.any(np.isnan(data[ch])):
                nan_mean = np.nanmean(data[ch], dtype=np.float32)
                data[ch] = np.nan_to_num(data[ch], nan=nan_mean).astype(np.float32)
        # ch, r, c = np.where(np.isnan(data))
        # data[ch, r, c] = np.nanmean(data[ch].astype(np.float32), dtype=np.float32)
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






def evaluate_on_all_sen_data_multi(model_number,
                                   country,
                                   use_veg_indices=True,
                                   num_per=1,
                                   save_inputs=False):
    with torch.no_grad():
        model_container = ModelTrainingContainer.load_container(model_number=model_number)
        country_dir = ppaths.waterway/f'country_data/{country}'
        inputs_dir = country_dir/'input_data'
        outputs_dir = country_dir/'output_data'
        for dir in [inputs_dir, outputs_dir]:
            if not dir.exists():
                dir.mkdir()
            else:
                delete_directory_contents(dir)
        sen_paths = list((country_dir/'sentinel_4326_cut').glob('*'))
        sen_paths = [path for path in sen_paths if (country_dir/f'elevation_cut/{path.name}').exists()]
        num_files = len(sen_paths)
        file_ind = 0
        s = tt()
        while file_ind <= num_files - 1:
            file_paths = []
            data_list = []
            s = tt()
            while len(file_paths) < num_per and file_ind <= num_files - 1:
                file = sen_paths[file_ind]
                # print(file)
                data, _, _ = open_all_and_merge(sentinel_path=file,
                                                country=country,
                                                make_indices=use_veg_indices
                                                )
                # print(file)
                data = fill_missing_data(data)
                if not data is None:
                    data = np.stack([data], axis=0)
                    if data is not None:
                        if save_inputs:
                            save_model_outputs(data, [file], inputs_dir)
                        file_paths.append(file)
                        data_list.append(data)
                file_ind += 1
                if file_ind%max(num_files//20, 1) == 0 and file_ind > num_files//20:
                    print(f'Completed {(file_ind + 1)/num_files:.2%} ({file_ind + 1}/{num_files})')
                    time_elapsed(s, 2)
            time_elapsed(s)
            data = np.concatenate(data_list, axis=0)
            data = torch.tensor(data, dtype=model_container.dtype, device=model_container.device)
            output = evaluate_model_on_single_image(model_container, data)
            save_model_outputs(output, file_paths, outputs_dir)
            del data
            del output
    for model in model_container.model_container.model_dict.values():
        model.to('cpu')
    del model_container
    torch.cuda.empty_cache()



def plot_results(country, input_file=None, file_index=None):
    country_dir = ppaths.waterway/f'country_data/{country}'
    input_files = list((country_dir/'input_data').glob('*'))
    output_dir = country_dir/'output_data'
    if input_file is None:
        input_file = input_files[file_index]
    output_file = output_dir/input_file.name
    # input_file = country_dir/f'sentinel_4326_cut/{input_file.name}'
    print(input_file)
    print(output_file)
    with rio.open(input_file) as rio_f_in:
        input_data = rio_f_in.read()
        print(input_data.shape)
        print(rio_f_in.bounds)
        sen_data = (input_data[1:4]+1)/2
        el_data = input_data[-1]
        w,s,e,n = rio_f_in.bounds

    with rio.open(output_file) as rio_f_out:
        output_data = rio_f_out.read()[0]
        print(rio_f_out.bounds)

    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
    ax[0].imshow(sen_data.transpose((1,2,0)), extent=(w,e,s,n))
    ax[1].imshow(el_data, extent=(w,e,s,n))
    ax[2].imshow(np.round(output_data), extent=(w,e,s,n))

def file_name_to_bbox(file_name):
    bbox = file_name.split('.tif')[0].split('_')[1:]
    bbox = [float(val) - .1*(-1)**(ind//2) for ind, val in enumerate(bbox)]
    return bbox

import shapely
def name_to_box(file_name):
    bbox = file_name_to_bbox(file_name)
    return shapely.box(*bbox)


def find_intersections(bbox, country, data_dir='input_data'):
    im_box = shapely.box(*bbox)
    elevation_dir = ppaths.waterway/f'country_data/{country}/{data_dir}'
    elevation_files = list(elevation_dir.glob('*'))
    intersection_files = []
    for elevation_file in elevation_files:
        el_box = name_to_box(elevation_file.name)
        if im_box.intersects(el_box):
            if el_box.contains(im_box):
                return [elevation_file]
            else:
                intersection_files.append(elevation_file)
    return intersection_files


if __name__ == '__main__':
    evaluate_on_all_sen_data_multi(245, country='ethiopia', use_veg_indices=True, num_per=16)
    # input_files = find_intersections((29.2, -1.8, 29.9, -1.3), 'rwanda')
    # ctry = 'rwanda'
    # index = 20
    # # print(len(input_files))
    # input_files = list((ppaths.waterway/f'country_data/{ctry}/input_data').glob('*'))
    # plot_results(input_file=input_files[index], country=ctry)
    # output_dir = ppaths.waterway/f'country_data/{ctry}/output_data'
    #
    # input_file = input_files[index]
    # output_file = output_dir/input_file.name


    # fig, ax = plt.subplots(1, 2)



    # evaluate_on_all_sen_data_multi(121, country='rwanda', use_veg_indices=True)
    # from pprint import pprint
    # paths = ppaths.waterway/'waterways_burned'
    # path = list((paths.glob('*')))[0]
    # rf = save_model_outputs(1, path, 10, 10)
    # pprint(rf.__dir__())
