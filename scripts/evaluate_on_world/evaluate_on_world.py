import os
from pathlib import Path
from water.make_country_waterways.run_polygon_eval import run_polygon_evaluate
from water.basic_functions import ppaths, delete_directory_contents, tt, time_elapsed
import geopandas as gpd
import shapely
from multiprocessing import Process
import numpy as np
import rasterio as rio


def copy_sentinel_from_storage(
        polygon: shapely.Polygon, sentinel_gdf: gpd.GeoDataFrame, sentinel_tree: shapely.STRtree
) -> None:
    file_indices = sentinel_tree.query(polygon, 'intersects')
    files_to_copy = sentinel_gdf.loc[file_indices]
    print(f"copying {len(files_to_copy)} files from storage", len(file_indices))
    temp_dir = ppaths.country_data/'sentinel_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = ppaths.storage_data/'sentinel_4326'
    sentinel_dir = ppaths.country_data/'sentinel_4326'
    for file in files_to_copy.file_name:
        if not (sentinel_dir/file).exists():
            os.system(f'cp {storage_dir/file} {temp_dir/file}')


def remove_current_sentinel_data(
        polygon: shapely.Polygon, sentinel_gdf: gpd.GeoDataFrame, sentinel_tree: shapely.STRtree
) -> None:
    files_to_keep = set(sentinel_gdf.loc[sentinel_tree.query(polygon, 'intersects')].file_name)
    sentinel_dir = ppaths.country_data/'sentinel_4326'
    for file in sentinel_dir.iterdir():
        if file.name not in files_to_keep:
            os.remove(file)


def move_temp_data_to_sentinel():
    sentinel_dir = ppaths.country_data/'sentinel_4326'
    temp_dir = ppaths.country_data/'sentinel_temp'
    for file in temp_dir.iterdir():
        file_name = file.name
        new_path = sentinel_dir/file_name
        file.rename(new_path)


def update_raster_datatype(file_path: Path, save_path: Path):
    with rio.open(file_path) as rio_f:
        profile = rio_f.profile
        dtype = profile['dtype']
        profile['dtype'] = 'uint8'
        if dtype != 'uint8':
            data = rio_f.read()
        else:
            data = np.zeros([1, 1], dtype=np.uint8)
    if data.dtype != np.uint8:
        data[data > 1] = 1
        data[data < 0] = 0
        data = np.ceil(data*255).astype(np.uint8)
        profile['nodata'] = 0
        print(len(data[(data != 255) & (data != 0)]))
        with rio.open(save_path, 'w', **profile) as dst:
            dst.write(data)
        # print(file_path)
    else:
        file_path.rename(save_path)


def run_process(
        save_dir_base: Path, name: str, polygon: shapely.Polygon,
        index: int, num_indices: int, evaluate_inputs: dict, copy_inputs: dict
):
    temp_dir = save_dir_base/'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir_base/f'{name}.tif'
    if not save_path.exists():
        print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f'\n  Working on {name} ({index + 1}/{num_indices})\n')
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
        s = tt()
        evaluate_inputs.update(
            {'polygon': polygon, 'polygon_name': name, 'save_dir_path': temp_dir}
        )
        evaluate_process = Process(target=run_polygon_evaluate, kwargs=evaluate_inputs)
        copy_process = Process(target=copy_sentinel_from_storage, kwargs=copy_inputs)
        evaluate_process.start()
        copy_process.start()
        copy_process.join()
        copy_process.close()
        evaluate_process.join()
        evaluate_process.close()
        remove_current_sentinel_data(**copy_inputs)
        move_temp_data_to_sentinel()
        output_path = temp_dir/f'output_data_merged/{name}_merged_data.tif'
        try:
            update_raster_datatype(output_path, save_path)
        except:
            print(f'Issue with {name}')
        delete_directory_contents(temp_dir)
        print('')
        time_elapsed(s, 4)


if __name__ == '__main__':
    s = tt()
    sentinel_storage_info = gpd.read_parquet(ppaths.storage_data/'sentinel_4326/sentinel_4326.parquet')
    sentinel_storage_tree = shapely.STRtree(sentinel_storage_info.geometry.to_numpy())
    polygon_info = gpd.read_parquet(ppaths.country_lookup_data/'zoom_level_6_xyz_info.parquet')

    # polygon_info = polygon_info[(polygon_info.x_index==37) & (polygon_info.y_index==32)]

    world_polygon = shapely.unary_union(
        gpd.read_parquet(ppaths.country_lookup_data/'world_boundaries.parquet').geometry.to_numpy()
    ).buffer(.15)
    time_elapsed(s)
    s = tt()
    polygon_info = polygon_info[polygon_info.intersects(world_polygon)].reset_index(drop=True)
    polygon_tree = shapely.STRtree(polygon_info.geometry.to_numpy())
    # polygon_info = polygon_info.loc[polygon_tree.query(world_polygon, predicate='contains')].reset_index(drop=True)
    from water.basic_functions import printdf
    polygon_info = polygon_info.loc[polygon_tree.query(world_polygon.boundary, predicate='intersects')].reset_index(drop=True)

    # polygon_info = polygon_info[(36 <= polygon_info.x_index) & (36 >= polygon_info.x_index)
    #                             & (polygon_info.y_index >= 32) & (polygon_info.y_index <= 34)]
    # polygon_info = polygon_info[(polygon_info.x_index == 36) & (polygon_info.y_index == 32)].reset_index(drop=True)
    # print(len(polygon_info))
    polygon_info = polygon_info.sort_values(by=['x_index', 'y_index']).reset_index(drop=True)
    time_elapsed(s)

    save_dir_base = ppaths.country_data/'model_outputs_zoom_level_6/'

    polygon_info['file_exists'] = polygon_info[['x_index', 'y_index']].apply(
        lambda row: (save_dir_base/f'{row.x_index}_{row.y_index}.tif').exists(), axis=1
    )
    polygon_info = polygon_info[~polygon_info.file_exists].reset_index(drop=True)


    # polygon_info.to_parquet(save_dir_base/'remaining_tiles.parquet')

    save_dir_base.mkdir(parents=True, exist_ok=True)
    model_num = 841
    eval_grid_width = 1024//2
    num_proc = 24
    recut_output = True
    output_width = 1024//2
    output_grid_res = 19.109
    num_per = 20*4

    eval_inputs = dict(
        eval_grid_width=eval_grid_width, eval_grid_step_size=eval_grid_width*3//4,
        model_number=model_num, num_proc=num_proc, num_per=num_per,
        recut_output_data=recut_output, output_grid_width=output_width, output_grid_res=output_grid_res,
        base_dir_path=ppaths.country_data
    )
    copy_inputs = dict(
        polygon=polygon_info.geometry[0], sentinel_gdf=sentinel_storage_info, sentinel_tree=sentinel_storage_tree
    )
    # s = tt()
    # inds = sentinel_storage_tree.query(polygon_info.geometry[0], predicate='intersects')
    # print(len(inds), len(sentinel_storage_info.loc[inds]))
    copy_sentinel_from_storage(**copy_inputs)
    remove_current_sentinel_data(**copy_inputs)
    move_temp_data_to_sentinel()
    time_elapsed(s)
    times = []
    # count = 0
    for row_index, x_index, y_index, polygon in zip(
            polygon_info.index, polygon_info.x_index, polygon_info.y_index, polygon_info.geometry
    ):
        s_i = tt()
        polygon = polygon.intersection(world_polygon)
        if row_index + 1 < len(polygon_info):
            next_polygon = polygon_info.geometry[row_index + 1]
            copy_inputs.update({'polygon': next_polygon})
        polygon_name = f'{x_index}_{y_index}'
        run_process(
            save_dir_base=save_dir_base, name=polygon_name, polygon=polygon, evaluate_inputs=eval_inputs,
            copy_inputs=copy_inputs, index=row_index, num_indices=len(polygon_info)
        )
        times.append(tt()-s_i)
        time_elapsed(s, 2)
        mean_time = int(np.mean(times))
        print(f'  Mean time per iteration : {mean_time//60}m, {mean_time%60}s')

