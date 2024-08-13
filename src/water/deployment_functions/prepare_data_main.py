import shapely

from water.deployment_functions.reference_grids import make_grids
from water.deployment_functions.cut_data import cut_data_to_match
from pathlib import Path


def prepare_main(save_dir: Path, polygon: shapely.Polygon, grid_width: int = 1280, grid_res: int=10, step_size: int = None,
                 **kwargs):
    if step_size is None:
        step_size = grid_width
    make_grids(
        save_dir=save_dir, polygon=polygon, grid_width=grid_width, grid_res=grid_res,
        step_size=step_size, **kwargs
    )
    # for data_dir_name in data_dir_names:
    #     data_dir = save_dir/f'{data_dir_name}'
    #     if not data_dir.exists():
    #         data_dir.mkdir()


if __name__ == '__main__':
    prepare_main('ethiopia', grid_width=1280, data_dir_names=('elevation', 'sentinel_4326',), num_proc=20)

    # prepare_main('zambia',grid_width=320)
