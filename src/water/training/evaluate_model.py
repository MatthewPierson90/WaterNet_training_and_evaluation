import os

import pandas as pd
import torch
# from water.training.model_containerV2 import ModelTrainingContainer
from water.training.model_container import ModelTrainingContainer

from water.data_functions.load.load_waterway_data import SenElBurnedLoaderEval, SenElTrailLoaderEval, SenElBurnedTrailLoader
from water.data_functions.clean.merge_waterway_data import ww_val_to_per
from water.loss_functions.loss_functions import WaterwayLossDecForEval, WaterwayTrailLossDecForEval
import numpy as np
import rasterio as rio
from water.basic_functions import (ppaths, my_pool, tt, time_elapsed, printdf,
                                   Path, delete_directory_contents
                                   )
from rasterio.transform import from_bounds
from multiprocessing import Process
import seaborn as sns
from water.data_functions.load.cut_data import cut_data_to_match_file_list
import shutil


def get_input_files(
        eval_dir=ppaths.waterway/'model_inputs_224',
        save_dir=ppaths.waterway/'model_inputs_224/output_data',
        input_dir_name='input_data',
        shuffle_files=False
):
    input_path = eval_dir/input_dir_name
    sub_dirs = input_path.glob('*')
    files = []
    for dir in sub_dirs:
        if not (save_dir/f'{dir.name}').exists():
            files.extend(dir.glob('*'))
    if shuffle_files:
        np.random.shuffle(files)
    else:
        files.sort(key=lambda x: x.name)
    return files


def check_input(file: Path, data_loader: SenElBurnedLoaderEval, side_len: int = None):
    data = data_loader.open_data(file)
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(file)
        delete_directory_contents(file)
        os.rmdir(file)
        if len(list(file.parent.glob('*'))) == 0:
            os.rmdir(file.parent)
    elif side_len is not None:
        if data.shape[-2] != side_len or data.shape[-1] != side_len:
            print(file)
            delete_directory_contents(file)
            os.rmdir(file)
            if len(list(file.parent.glob('*'))) == 0:
                os.rmdir(file.parent)

def check_input_list(files: list, data_loader: SenElBurnedLoaderEval, side_len: int):
    for file in files:
        check_input(file, data_loader, side_len=side_len)


def clean_inputs(
        data_loader: SenElBurnedLoaderEval,
        num_proc: int = 15, base_dir: Path = ppaths.waterway/'model_inputs_224',
        side_len: int = None,
):
    file_list = get_input_files(base_dir)
    step_size = len(file_list)//(num_proc*8)
    num_steps = int(np.ceil(len(file_list)/step_size))
    inputs = [{
        'files': file_list[step_size*i: step_size*(i + 1)],
        'data_loader': data_loader, 'side_len': side_len,
    } for i in range(num_steps)]
    # print(step_size*num_steps, len(file_list))
    my_pool(func=check_input_list, input_list=inputs, num_proc=num_proc, use_kwargs=True)


def save_model_outputs(model_output, file_paths, save_dir):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    for file_ind, file_path in enumerate(file_paths):
        save_path_parent = save_dir/file_path.parent.name
        if not save_path_parent.exists():
            save_path_parent.mkdir()
        save_path = save_path_parent/file_path.name
        with rio.open(file_path/'sentinel.tif') as rio_f:
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
        # return rio_f
        with rio.open(save_path, 'w', **save_profile) as dst_f:
            dst_f.write(model_output[file_ind])


def evaluate_model_on_single_image(
        model_container: ModelTrainingContainer, data, max_per_it, num_y=1
):
    num_its = int(np.ceil(len(data)/max_per_it))
    output_cpu = None
    for i in range(num_its):
        x = torch.tensor(
            data[i*max_per_it:(i+1)*max_per_it, :-num_y], device=model_container.device, dtype=model_container.dtype
        )
        y = torch.tensor(
            data[i*max_per_it:(i+1)*max_per_it, -num_y:], device=model_container.device, dtype=model_container.dtype
        )
        output = model_container.evaluate_model(model_name='wwm', train=False, input=x)
        model_container.evaluate_loss_function(model_name='wwm', inputs=output, targets=y)
        if output_cpu is None:
            output_cpu = output.detach().to('cpu').float()
        else:
            out = output.detach().to('cpu').float()
            output_cpu = np.concatenate([output_cpu, out], axis=0)
        del x
        del y
        del output
    return output_cpu


def copy_burned_files(file_paths, dst_dir):
    for file_path in file_paths:
        save_dir = dst_dir/file_path.name
        if not save_dir.exists():
            save_dir.mkdir()
        shutil.copytree(file_path, save_dir, dirs_exist_ok=True)


def mk_dirs(file_paths, dst_dir):
    for file_path in file_paths:
        save_dir = dst_dir/file_path.name
        if not save_dir.exists():
            save_dir.mkdir()


def cut_next_file_set(file_paths, base_dir_path, num_proc=15):
    delete_directory_contents(base_dir_path)
    mk_dirs(file_paths, base_dir_path)
    num_per = int(np.ceil(len(file_paths) / num_proc))
    # input_list = [
    #     dict(file_paths=file_paths[i: (i+1)*num_per], dst_dir=base_dir_path) for i in range(num_proc)
    # ]
    # my_pool(num_proc=num_proc, func=copy_burned_files, input_list=input_list, use_kwargs=True, print_progress=False)
    # cut_data_to_match_file_list(
    #     data_dir=ppaths.waterway/'waterways_burned', file_paths=file_paths,
    #     base_dir_path=base_dir_path, num_proc=num_proc
    # )
    cut_data_to_match_file_list(
        data_dir=ppaths.waterway/'elevation_cut', file_paths=file_paths, base_dir_path=base_dir_path, num_proc=num_proc
    )


def make_save_next_temp_file(input_list: list,
                             data_loader: SenElBurnedLoaderEval,
                             temp_dir: Path,
                             temp_cut_dir: Path,
                             num_proc: int=15,
                             temp_name: str='next_input.npy'
                             ):
    data = []
    s = tt()
    cut_next_file_set(file_paths=input_list, base_dir_path=temp_cut_dir, num_proc=num_proc)
    # load_file_paths = temp_cut_dir
    for file_path in input_list:
        loaded = data_loader.open_data(file_path)
        if np.any(np.isnan(loaded)) or np.any(np.isinf(loaded)):
            print(file_path)
            print(np.where(np.isnan(loaded)))
            print(np.where(np.isinf(loaded)))
        else:
            data.append(loaded)

    try:
        data = np.stack(data, axis=0, dtype=np.float32)
        np.save(temp_dir / temp_name, data)
    except ValueError:
        for item in data:
            print(item.shape)
        raise Exception


def time_delta(s):
    return tt()-s


def evaluate_on_all_sen_data_multi(
        model_number: int, evaluation_dir: Path = ppaths.waterway/'model_inputs_224',
        num_per_load: int = 1650, max_per_it: int = 550, max_target: int = 21, num_y=1,
        data_loader: SenElBurnedLoaderEval = None, is_terminal=True, input_dir_name='input_data',
        output_dir_name='output_data', stats_name='evaluation_stats', loss_func_type=WaterwayLossDecForEval
):
    temp_dir: Path = evaluation_dir/'temp'
    temp_cut_dir = evaluation_dir/'temp_cut'
    save_dir_path = evaluation_dir/f'{output_dir_name}_{model_number}'
    save_stats_path = evaluation_dir/f'{stats_name}_{model_number}.parquet'
    if data_loader is None:
        data_loader = SenElBurnedLoaderEval(el_base=temp_cut_dir)
    if not temp_dir.exists():
        temp_dir.mkdir()
    delete_directory_contents(temp_dir)
    with torch.no_grad():
        model_container = ModelTrainingContainer.load_container(model_number=model_number)
        to_save = {'file_name': []}
        model_container.clear_all_model_loss_list_dicts()
        num_factors = model_container.model_container.loss_dict['wwm'].num_factors
        loss_func = loss_func_type(num_factors=num_factors, max_target=max_target)
        pixel_stats = loss_func.pixel_stats
        model_container.model_container.loss_dict['wwm'] = loss_func
        lld = model_container.model_container.loss_dict.get_model_loss_list_dict('wwm')
        for key, val in lld.items():
            to_save[key] = val
        if save_stats_path.exists():
            df = pd.read_parquet(save_stats_path)
            for col in df.columns:
                if col in to_save:
                    to_save[col].extend(df[col])
        files = get_input_files(
            evaluation_dir, save_dir=save_dir_path, input_dir_name=input_dir_name, shuffle_files=True
        )
        num_files = len(files)
        file_inputs = files[:num_per_load]
        make_save_next_temp_file(
                input_list=file_inputs, temp_dir=temp_dir, data_loader=data_loader, temp_cut_dir=temp_cut_dir
        )
        file_ind = num_per_load
        # s = tt()
        next_print = 0
        print_divide = max(num_files//200, 1)
        start = tt()
        st = tt()
        start_temp = 0
        load = 0
        eval_save = 0
        finish_temp = 0
        while file_ind <= num_files - 1:
            s = tt()
            file_paths = file_inputs
            to_save['file_name'].extend([f'{fp.parent.name}/{fp.name}' for fp in file_paths])
            file_inputs = files[file_ind:file_ind + num_per_load]
            file_ind += num_per_load
            kwargs = {
                'input_list': file_inputs, 'data_loader': data_loader,
                'temp_dir': temp_dir, 'temp_cut_dir':  temp_cut_dir
            }
            temp_process = Process(target=make_save_next_temp_file, kwargs=kwargs)
            temp_process.start()
            start_temp += time_delta(s)
            s = tt()
            data = np.load(temp_dir/'next_input.npy')
            load += time_delta(s)
            s = tt()
            output = evaluate_model_on_single_image(model_container, data, max_per_it=max_per_it, num_y=num_y)
            save_model_outputs(output, file_paths, save_dir=save_dir_path)
            eval_save += time_delta(s)
            if file_ind//print_divide >= next_print:
                if is_terminal:
                    os.system('clear')
                next_print = 1 + file_ind//print_divide
                print(f'Completed {(file_ind)/num_files:.2%} ({file_ind}/{num_files})')
                time_elapsed(st,2)
                st = tt()
                for key, val in lld.items():
                    if 'target' not in key and 'num' not in key:
                        print(f'{key:>12}: {np.nanmean(np.array(val)):.4f}')
                print('')
                for key, val in pixel_stats.items():
                    print(f'{key:>12}: {val:.4f}')
                print('')
                time_elapsed(start)
                print(f'start temp: {start_temp:.2f}')
                print(f'      load: {load:.2f}')
                print(f' eval save: {eval_save:.2f}')
                print(f'  fin temp: {finish_temp:.2f}')
                start_temp, load, eval_save, finish_temp = 0, 0, 0, 0
            df = pd.DataFrame(to_save)
            df.to_parquet(save_stats_path)
            s = tt()
            temp_process.join()
            temp_process.close()
            finish_temp += time_delta(s)

        file_paths = file_inputs
        to_save['file_name'].extend([f'{fp.parent.name}/{fp.name}' for fp in file_paths])
        data = np.load(temp_dir / 'next_input.npy')
        output = evaluate_model_on_single_image(model_container, data, max_per_it=max_per_it, num_y=num_y)
        save_model_outputs(output, file_paths, save_dir=save_dir_path)
        df = pd.DataFrame(to_save)
        # df.to_csv(ppaths.waterway/evaluation_dir/'evaluation_stats.csv')
        df.to_parquet(save_stats_path)


if __name__ == '__main__':
    import geopandas as gpd
    import seaborn as sns
#
    # fils = get_input_files(ppaths.waterway/'model_inputs_224')
    # print(fils[0])

    model_number = 822
    max_per_it = 32
    kwargs = dict(
        model_number=model_number, evaluation_dir=ppaths.waterway/'model_inputs_832',
        num_per_load=2*max_per_it, max_per_it=max_per_it,
        data_loader=SenElBurnedLoaderEval(
            el_base=ppaths.waterway/'model_inputs_832/temp_cut', elevation_name='elevation_cut'
        ), max_target=[22], input_dir_name='input_data', output_dir_name='output_data', stats_name='evaluation_stats',
        loss_func_type=WaterwayTrailLossDecForEval, num_y=1
    )
    p = Process(target=evaluate_on_all_sen_data_multi, kwargs=kwargs)
    p.start()
    p.join()
    p.close()

    kwargs = dict(
        model_number=model_number, evaluation_dir=ppaths.waterway/'model_inputs_832',
        num_per_load=max_per_it*2, max_per_it=max_per_it,
        data_loader=SenElBurnedLoaderEval(
            el_base=ppaths.waterway/'model_inputs_832/temp_cut', elevation_name='elevation_cut'
        ), max_target=[22], input_dir_name='test_data', output_dir_name='output_test_data', stats_name='evaluation_test',
        loss_func_type=WaterwayTrailLossDecForEval, num_y=1
    )
    p = Process(target=evaluate_on_all_sen_data_multi, kwargs=kwargs)
    p.start()
    p.join()
    p.close()

    kwargs = dict(
        model_number=model_number, evaluation_dir=ppaths.waterway/'model_inputs_832',
        num_per_load=max_per_it*2, max_per_it=max_per_it,
        data_loader=SenElBurnedLoaderEval(
            el_base=ppaths.waterway/'model_inputs_832/temp_cut', elevation_name='elevation_cut'
        ), max_target=[22], input_dir_name='val_data', output_dir_name='output_val_data', stats_name='evaluation_val',
        loss_func_type=WaterwayTrailLossDecForEval, num_y=1
    )
    p = Process(target=evaluate_on_all_sen_data_multi, kwargs=kwargs)
    p.start()
    p.join()
    p.close()


    # model_number = 775
    # max_per_it = 32
    # kwargs = dict(
    #     model_number=model_number, evaluation_dir=ppaths.waterway/'model_inputs_832',
    #     num_per_load=2*max_per_it, max_per_it=max_per_it,
    #     data_loader=SenElBurnedTrailLoader(
    #         el_base=ppaths.waterway/'model_inputs_832/temp_cut', elevation_name='elevation_cut'
    #     ), max_target=[21, 2], input_dir_name='input_data', output_dir_name='output_data', stats_name='evaluation_stats',
    #     loss_func_type=WaterwayTrailLossDecForEval, num_y=2
    # )
    # p = Process(target=evaluate_on_all_sen_data_multi, kwargs=kwargs)
    # p.start()
    # p.join()
    # p.close()
    #
    # kwargs = dict(
    #     model_number=model_number, evaluation_dir=ppaths.waterway/'model_inputs_832',
    #     num_per_load=max_per_it*2, max_per_it=max_per_it,
    #     data_loader=SenElBurnedTrailLoader(
    #         el_base=ppaths.waterway/'model_inputs_832/temp_cut', elevation_name='elevation_cut'
    #     ), max_target=[21, 2], input_dir_name='test_data', output_dir_name='output_test_data', stats_name='evaluation_test',
    #     loss_func_type=WaterwayTrailLossDecForEval, num_y=2
    # )
    # p = Process(target=evaluate_on_all_sen_data_multi, kwargs=kwargs)
    # p.start()
    # p.join()
    # p.close()
    #
    # kwargs = dict(
    #     model_number=model_number, evaluation_dir=ppaths.waterway/'model_inputs_832',
    #     num_per_load=max_per_it*2, max_per_it=max_per_it,
    #     data_loader=SenElBurnedTrailLoader(
    #         el_base=ppaths.waterway/'model_inputs_832/temp_cut', elevation_name='elevation_cut'
    #     ), max_target=[21, 2], input_dir_name='val_data', output_dir_name='output_val_data', stats_name='evaluation_val',
    #     loss_func_type=WaterwayTrailLossDecForEval, num_y=2
    # )
    # p = Process(target=evaluate_on_all_sen_data_multi, kwargs=kwargs)
    # p.start()
    # p.join()
    # p.close()
    #







def compare_model_describe(df_list, df_names=None, percentile=10):
    if df_names is None:
        df_names = [f'model_{i}' for i in range(len(df_list))]
    percentiles = [round(i/percentile, 2) for i in range(1, percentile)]
    for df, name in zip(df_list, df_names):
        print(name)
        printdf(df.describe(percentiles), 1000000)
        print('')


def compare_model_plots(df_list, df_names=None, column='f1'):
    if df_names is None:
        df_names = [f'model_{i}' for i in range(len(df_list))]
    for df, name in zip(df_list, df_names):
        df['model'] = name
    df = pd.concat(df_list, ignore_index=True)
    sns.displot(data=df, x=column, hue='model')


def get_inds(df):
    index = df[~(
            (
                    (df.precision < .5) & (df.recall < .5)
            )
            |
            (
                    (df.accuracy < .75) & ((df.recall < .75) | (df.precision > .25))
            )
    )].index
    return index



# if __name__ == '__main__':
#     import geopandas as gpd
#     import seaborn as sns
#     evaluate_on_all_sen_data_multi(model_number=150, num_per=500)
