from water.basic_functions import ppaths, Path, my_pool, delete_directory_contents, printdf
import rasterio as rio
import numpy as np
import pandas as pd
import shapely
import geopandas as gpd


def get_data_types(file_path: Path, file_type: str):
    with rio.open(file_path/file_type) as rio_f:
        data = rio_f.read()
    values, counts = np.unique(data, return_counts=True)
    to_return = {value: count for value, count in zip(values, counts)}
    return to_return


def make_file_data_dict(
        file_path: Path, file_type: str, col_name_dict: dict
):
    to_return = {}
    to_return['file_name'] = f'{file_path.parent.name}/{file_path.name}'
    value_count_dict = get_data_types(file_path, file_type)
    max_value = max(list(col_name_dict.keys()))
    for value in range(max_value+1):
        to_return[col_name_dict[value]] = value_count_dict.get(value, 0)
    return to_return


def make_file_list_dataframe(
        dir_list: list[Path], file_type: str, col_name_dict: dict, save_path: Path
):
    data_list = []
    for sub_dir in dir_list:
        for file_path in sub_dir.iterdir():
            data_list.append(make_file_data_dict(file_path, file_type, col_name_dict))
    df = pd.DataFrame(data_list)
    df.to_parquet(save_path)
    return df


def get_index_dir_paths(index: int):
    base_dir = ppaths.waterway/f'model_inputs_{index}/input_data'
    file_paths = list(base_dir.iterdir())
    return file_paths



def concatenate_temp_dfs(temp_dir: Path):
    df_list = []
    for df_path in temp_dir.iterdir():
        df_list.append(pd.read_parquet(df_path))
    df = pd.concat(df_list, ignore_index=True)
    return df


def make_file_dataframe_multi(
        index: int, file_type: str, col_name_dict: dict, num_proc: int=16, save_name: str=None
):
    if save_name is None:
        save_name = f'{file_type.split(".tif")[0]}_data_types.parquet'
    dir_paths = get_index_dir_paths(index)
    base_dir = ppaths.waterway/f'model_inputs_{index}'
    temp_dir = base_dir/'temp_dataframes'
    if not temp_dir.exists():
        temp_dir.mkdir()
    if num_proc > 1:
        step_size = len(dir_paths)//(2*num_proc-1)
        input_list = [{
            'dir_list': dir_paths[step_size*i: step_size*(i+1)],
            'file_type': file_type,
            'col_name_dict': col_name_dict,
            'save_path': temp_dir/f'temp_df_{i}.parquet'
        } for i in range(60*num_proc)]
        my_pool(num_proc=num_proc, input_list=input_list, func=make_file_list_dataframe, use_kwargs=True)
        df = concatenate_temp_dfs(temp_dir)
        df.to_parquet(base_dir/save_name)
        delete_directory_contents(temp_dir)
        temp_dir.rmdir()
    else:
        make_file_list_dataframe(
            dir_list=dir_paths, file_type=file_type, col_name_dict=col_name_dict, save_path=base_dir/save_name
        )


waterways_burned_col_names = {
    0: 'land',
    1: 'playa',
    2: 'inundation',
    3: 'swamp_i',
    4: 'swamp_p',
    5: 'swamp',
    6: 'reservoir',
    7: 'lake_i',
    8: 'lake_p',
    9: 'lake',
    10: 'spillway',
    11: 'drainage',
    12: 'wash',
    13: 'canal_storm',
    14: 'canal_aqua',
    15: 'canal',
    16: 'artificial_path',
    17: 'ephemeral',
    18: 'intermittent',
    19: 'perennial',
    20: 'streams',
    21: 'other',
}


def name_to_shapely_box(file_name):
    box_info = [float(item) for item in file_name.split('/')[0].split('bbox_')[-1].split('_')]
    return shapely.box(*box_info)


def add_hu4_index(df):
    hulls_df = gpd.read_parquet(ppaths.waterway/'hu4_hulls.parquet')
    df['geometry'] = df.file_name.apply(name_to_shapely_box)
    df = gpd.GeoDataFrame(df, crs=4326).reset_index(drop=True)
    df = df.reset_index()
    str_tree = shapely.STRtree(df.geometry.to_list())
    df_idx_to_hull = {}
    for idx, geom in zip(hulls_df.hu4_index, hulls_df.geometry):
        df_idxs = str_tree.query(geom, predicate='contains')
        for df_idx in df_idxs:
            df_idx_to_hull[df_idx] = idx
    df['hu4_index'] = df['index'].apply(lambda x: df_idx_to_hull[x])
    return df

if __name__ == '__main__':
    index = 224
    # make_file_dataframe_multi(
    #     index=832, file_type='waterways_burned.tif', col_name_dict=waterways_burned_col_names, num_proc=24
    # )
    pd.options.display.float_format = '{:.6f}'.format
    df = pd.read_parquet(ppaths.waterway/f'model_inputs_{index}/waterways_burned_data_types.parquet')
    for col in df.columns:
        if col != 'file_name':
            df[col] = df[col]/index**2
    # df = add_hu4_index(df)
    # df_group = df.groupby('hu4_index').mean(numeric_only=True)
    # printdf(df_group, 200)
    # printdf(df.describe(.1*i for i in range(0, 10)), 200)
    # df1 = df[(
    #         (
    #                 (df.land == 1) | ((0 < df.swamp_i) & (df.swamp_i < .3)) | ((0 < df.swamp_p) & (df.swamp_p < .3)) |
    #                 (.05 < df.lake_p) | (0 < df.ephemeral) | (0 < df.intermittent)
    #         )
    #         &
    #         (
    #                 (df.canal < .2) & (df.playa < .2) & (df.reservoir < .2) &
    #                 (df.other < .2) & (df.swamp < .2) & (df.inundation < .2)
    #         )
    #           )
    # ]
    # df1 = df[(
    #         (
    #                 (df.swamp_i < .05) | (df.swamp_p < .05) |
    #                 (df.swamp < .05)
    #                 # | (0 < df.ephemeral) | (.03 < df.intermittent) | (0 < df.wash)
    #         )
    #         &
    #         (
    #                 (df.canal < .2) & (df.playa == 0) & (df.reservoir < .2) & (df.other < .2) & (df.inundation < .2)
    #         )
    #           )
    # ]
    df1 = df[
            (
                    (df.swamp_i == 0) & (df.swamp_p == 0) & (df.swamp < .001) & (df.lake_i == 0) & (df.drainage == 0)
                    & (df.playa == 0) & (df.inundation == 0)
            )
    ]
    # df2 = df[
    #         ~(
    #                 (df.swamp_i == 0) & (df.swamp_p == 0) & (df.swamp < .02) & (df.lake_i == 0) & (df.drainage == 0)
    #                 & (df.playa == 0) & (df.inundation == 0)
    #         )
    # ]
    # print(len(df), len(df1), 1-len(df1)/len(df))
    print(len(df1)/len(df))
    # for value in waterways_burned_col_names.values():
    #     print(value, len(df1[df1[value] > 0]), len(df1[df1[value]>0]) / len(df[df[value]>0]), len(df2[df2[value] > 0]), len(df2[df2[value] > 0])/len(df[df[value]>0]))
    # print(df)
    # printdf(df)
    df1.to_parquet(ppaths.waterway/f'model_inputs_{index}/new_inputs.parquet')
    # print(df.sum())