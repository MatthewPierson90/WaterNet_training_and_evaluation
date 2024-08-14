
import numpy as np
import rasterio as rio
from water.basic_functions import SharedMemoryPool
from water.paths import ppaths
from rasterio.transform import from_bounds
from pathlib import Path


class DataGrid:
    def __init__(
            self, data: np.ndarray, missing_vals: list = (np.nan, np.inf, -np.inf), data_name=''
    ):
        self.is_empty = False
        if data.shape[0] == 0:
            self.is_empty = True
        if len(data.shape) == 2:
            data = np.stack([data], axis=0)
        elif len(data.shape) > 3:
            raise Exception('invalid data shape')
        self.data_name = data_name
        self.data = data.copy()
        self.missing_vals = missing_vals
        self.num_rows, self.num_cols = 0, 0
        if not self.is_empty:
            self.num_rows, self.num_cols = data.shape[1], data.shape[2]
        self.start_row, self.start_col = 0, 0
    
    def has_missing_data(self):
        for val in self.missing_vals:
            if np.any(self.data == val):
                sum_val = (self.data == val).sum()
                total = self.data.shape[0]*self.data.shape[1]*self.data.shape[2]
                print(sum_val/total)
                if sum_val/total < .1:
                    return False
                return True
        return False
    
    def get_slice(self, row: int, col: int, height: int, width: int):
        return DataSlice(
                data_grid=self, start_row=row, start_col=col, height=height, width=width
        )
    
    @classmethod
    def from_file_name(cls, file_name: str, parent_dir: Path):
        if (parent_dir/file_name).exists():
            with rio.open(parent_dir/file_name) as rio_f:
                data = rio_f.read()
                data_name = parent_dir.name
                missing_values = [rio_f.nodata, np.nan, np.inf, -np.inf]
            return DataGrid(data=data, data_name=data_name, missing_vals=missing_values)
        else:
            return DataGrid(np.array([]), [], '')


class DataSlice(DataGrid):
    def __init__(self, data_grid: DataGrid,
                 start_row: int, start_col: int,
                 height: int, width: int
                 ):
        self.parent_grid = data_grid
        data = data_grid.data[:, start_row: start_row + height, start_col: start_col + width]
        missing_vals = data_grid.missing_vals
        if data.shape[-1] != width or data.shape[-2] != height:
            data = data.astype(np.float32)
            data[0:1] = np.nan
        super().__init__(data=data, missing_vals=missing_vals, data_name=data_grid.data_name)
        self.start_row = start_row + self.parent_grid.start_row
        self.start_col = start_col + self.parent_grid.start_col


class DataGrids:
    def __init__(self, data_grids: list[DataGrid]):
        self.data_grids = data_grids
        self.is_empty = np.any([data_grid.is_empty for data_grid in data_grids])
        self.num_rows, self.num_cols = data_grids[0].num_rows, data_grids[0].num_cols
        self.start_row, self.start_col = 0, 0
    
    def has_missing_data(self):
        if np.any([data_grid.has_missing_data() for data_grid in self.data_grids]):
            return True
        else:
            return False
    
    def get_slices(self, start_row, start_col, height, width):
        return DataSlices(self, start_row, start_col, height, width)
    
    @classmethod
    def from_file_name(cls, file_name: str, parent_dirs: list[Path]):
        data_grids = []
        for parent_dir in parent_dirs:
            data_grids.append(DataGrid.from_file_name(file_name, parent_dir))
        return DataGrids(data_grids)


class DataSlices(DataGrids):
    def __init__(self, data_grids: DataGrids,
                 start_row: int, start_col: int,
                 height: int, width: int
                 ):
        self.parent_grids = data_grids
        slices = [dg.get_slice(start_row, start_col, height, width) for dg in data_grids.data_grids]
        super().__init__(data_grids=slices)
        self.start_row = start_row + self.parent_grids.start_row
        self.start_col = start_col + self.parent_grids.start_col


class DataSlicer(DataGrids):
    def __init__(self, data_grids: DataGrids,
                 slice_width: int, slice_height: int = None,
                 row_step_size: int = None, col_step_size: int = None,
                 missing_data_step_size: int = 25,
                 start_row: int = 0, start_col: int = 0):
        super().__init__(data_grids.data_grids)
        slice_height = slice_height if slice_height is not None else slice_width
        row_step_size = row_step_size if row_step_size is not None else slice_height
        col_step_size = col_step_size if col_step_size is not None else slice_width
        self.start_col = start_col
        self.current_row, self.current_col = start_row, start_col
        self.height, self.width = slice_height, slice_width
        self.row_step_size, self.col_step_size = row_step_size, col_step_size
        self.missing_data_step_size = missing_data_step_size
        self._missing_steps = 1
        self._complete = False
        self.slices = []
    
    def get_valid_slices(self):
        slice = self.get_slices(self.current_row, self.current_col, self.height, self.width)
        if slice.has_missing_data():
            self.iterate_row_col(from_missing=True)
        else:
            self.slices.append(slice)
            self.iterate_row_col()
        if not self._complete:
            self.get_valid_slices()
    
    def iterate_row_col(self, from_missing=False):
        if from_missing:
            col_step = self.missing_data_step_size*self._missing_steps
            row_step = col_step
            self._missing_steps += 1
        else:
            row_step = 0
            col_step = self.col_step_size
            self._missing_steps = 1
        next_col, next_row = self.current_col + col_step, self.current_row + row_step
        if next_col + self.width > self.num_cols:
            next_col = self.start_col
            next_row = next_row + self.row_step_size
        if next_row + self.height > self.num_rows:
            self._complete = True
        self.current_row, self.current_col = next_row, next_col

    @classmethod
    def from_file(cls, file_name: str,
                  parent_dirs: list[Path],
                  slice_width: int, **kwargs):
        data_grids = super().from_file_name(file_name, parent_dirs)
        return DataSlicer(data_grids=data_grids, slice_width=slice_width, **kwargs)


def save_data_slice(data_slice: DataSlice, save_dir_path: Path, file_path: Path):
    data = data_slice.data
    start_r, start_c = data_slice.start_row, data_slice.start_col
    with rio.open(file_path) as rio_f:
        save_profile = rio_f.profile
        count, height, width = data.shape
        x_res, y_res = rio_f.res
        x_res, y_res = abs(x_res), abs(y_res)
        w, n = rio_f.xy(start_r, start_c)
        e, s = rio_f.xy(start_r + height - 1, start_c + width - 1)
        w = w - x_res/2
        s = s - y_res/2
        e = e + x_res/2
        n = n + y_res/2
        transform = from_bounds(
                west=w, south=s, east=e, north=n, width=width, height=height
        )
        save_profile['transform'] = transform
        save_profile['height'] = height
        save_profile['width'] = width
    save_dir = save_dir_path/f'bbox_{w}_{s}_{e}_{n}'
    if not save_dir.exists():
        save_dir.mkdir()
    save_path = save_dir/f'{data_slice.data_name}.tif'
    save_profile['count'] = count
    save_profile['dtype'] = data.dtype
    save_profile['nodata'] = data_slice.missing_vals[0]
    with rio.open(save_path, 'w', **save_profile) as dst_f:
        dst_f.write(data)



def save_data_slices(data_slices: DataSlices, save_dir_path: Path,
                     file_path: Path, slice_index: int
                     ):
    save_dir_path = save_dir_path
    for data_slice in data_slices.data_grids:
        save_data_slice(data_slice, save_dir_path=save_dir_path, file_path=file_path)


def slice_and_save_data(file: Path, parent_dirs: list[Path], slice_width,
                        save_dir_path: Path, **kwargs
                        ):
    
    data_slicer = DataSlicer.from_file(
            file.name, parent_dirs=parent_dirs, slice_width=slice_width, **kwargs
    )
    if not data_slicer.is_empty:
        data_slicer.get_valid_slices()
        if not len(data_slicer.slices) == 0:
            dir_name = file.name.split('.tif')[0]
            save_dir_path = save_dir_path/dir_name
            if not save_dir_path.exists():
                save_dir_path.mkdir()
            if len(data_slicer.slices) == 0:
                print(file.name)
            for slice_ind, data_slices in enumerate(data_slicer.slices):
                save_data_slices(
                        data_slices, save_dir_path=save_dir_path, file_path=file, slice_index=slice_ind
                )


def save_inputs_multi(parent_dirs: list,
                      slice_width: int,
                      save_dir_path: Path = ppaths.training_data/'model_inputs',
                      num_proc=12, **kwargs
                      ):
    save_dir_path = save_dir_path/'input_data'
    if not save_dir_path.exists():
        save_dir_path.mkdir(parents=True)
    base_dir = parent_dirs[0]
    input_list = []
    for file in base_dir.glob('*'):
        inputs = {
            'file': file, 'parent_dirs': parent_dirs, 'slice_width': slice_width, 'save_dir_path': save_dir_path,
        }
        inputs.update(**kwargs)
        input_list.append(inputs)
    SharedMemoryPool(func=slice_and_save_data, input_list=input_list, use_kwargs=True, num_proc=num_proc).run()


def slice_and_save_list_data(files: list[Path], parent_dirs: list[Path], slice_width,
                            save_dir_path: Path, **kwargs
                            ):
    for file in files:
        slice_and_save_data(file, parent_dirs, slice_width, save_dir_path, **kwargs)


def check_files(file_list, dirs_to_check):
    new_list = []
    for file in file_list:
        for dir in dirs_to_check:
            if not (dir/file.name).exists():
                continue
        else:
            new_list.append(file)
    return new_list


def make_training_data_multi(
        parent_dirs: list,
        slice_width: int,
        save_dir_path: Path = ppaths.training_data/'model_inputs',
        dirs_to_check: list[Path] = (
                ppaths.elevation_cut, ppaths.sentinel_cut, ppaths.waterways_burned,
        ),
        num_proc=12,
        **kwargs
):
    save_dir_path = save_dir_path/'input_data'
    if not save_dir_path.exists():
        save_dir_path.mkdir(parents=True)
    base_dir = parent_dirs[0]
    input_list = []
    files = list(base_dir.glob('*'))
    files = check_files(files, dirs_to_check=dirs_to_check)
    files = [file for file in files]

    np.random.shuffle(files)
    num_files = len(files)
    step_size = max(num_files // (num_proc * 4), 1)
    num_steps = num_proc * 4
    for i in range(num_steps+1):
        inputs = {
            'files': files[i * step_size:(i + 1) * step_size],
            'parent_dirs': parent_dirs,
            'slice_width': slice_width,
            'save_dir_path': save_dir_path,
        }
        inputs.update(**kwargs)
        input_list.append(inputs)
    SharedMemoryPool(func=slice_and_save_list_data, input_list=input_list, use_kwargs=True, num_proc=num_proc).run()
