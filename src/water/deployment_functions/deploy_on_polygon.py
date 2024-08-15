import shapely
from water.deployment_functions.predict_data_stream import predict_on_all_sen_data_multi
from water.data_functions.prepare.add_weight import add_weight_to_all_outputs
from water.data_functions.prepare.reference_grids import make_reference_grids
from water.data_functions.prepare.cut_data import cut_data_to_match, merge_dir_and_save
from water.basic_functions import resuffix_directory_and_make_new
from water.paths import ppaths, Path
from water.basic_functions import tt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def deploy_on_polygon(
        polygon: shapely.Polygon or shapely.MultiPolygon,
        polygon_name: str, save_dir_path: Path,
        num_proc: int, model_number: int, num_per: int,  input_grid_width: int = 832,
        output_grid_width: int = None, base_dir_path: Path = ppaths.deploy_data,
        recut_output_data=False, output_grid_res=40,
        output_dir_name='output_data', eval_grid_step_size: int = None, **kwargs
):
    """

    Parameters
    ----------
    polygon: shapely.Polygon or shapely.MultiPolygon - The polygon to run the model on.
    polygon_name: str - The name of the polygon, this
    save_dir_path: Path - The path to the directory where you want to save the model.
    num_proc: int - The number of processors to use to cut the input data for the model.
    model_number: int - The model to deploy.
    num_per: int - The number of images per evaluation.
    input_grid_width
    output_grid_width
    base_dir_path
    recut_output_data
    output_grid_res
    output_dir_name
    eval_grid_step_size
    kwargs

    """
    if not save_dir_path.exists():
        save_dir_path.mkdir()
    output_grid_width = output_grid_width if output_grid_width is not None else input_grid_width
    eval_grid_step_size = input_grid_width if eval_grid_step_size is None else eval_grid_step_size
    print(f'Making reference grids for {polygon_name}')
    make_reference_grids(
        save_dir=save_dir_path/'temp', polygon=polygon, grid_width=input_grid_width, grid_res=10,
        step_size=eval_grid_step_size
    )
    print('\n')
    print(f'Running on {polygon_name}')
    predict_on_all_sen_data_multi(
        model_number=model_number, num_per=num_per, base_dir_path=base_dir_path, polygon_dir=save_dir_path/polygon_name,
        num_proc=num_proc, output_name=output_dir_name, input_grid_width=input_grid_width, **kwargs
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
        make_reference_grids(
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
