import shutil

import pandas as pd
from water.basic_functions import save_pickle, open_pickle, save_yaml, open_yaml, delete_directory_contents
from water.paths import ppaths
from water.data_functions.load.cut_data import cut_data_to_match_file_list
import numpy as np
import shapely
import rasterio as rio
from multiprocessing import Process
from pathlib import Path
import importlib


class DataOpener:
    def __init__(self, file_type: str):
        self.file_type = file_type

    def _open_data(self, base_path: Path):
        with rio.open(base_path / f'{self.file_type}.tif') as rio_f:
            data = rio_f.read()
        return data


class VegIndexComputer:
    def compute(self, sen_data: np.ndarray):
        pass


class NDWIComputer(VegIndexComputer):
    def compute(self, sen_data):
        return (sen_data[2:3] - sen_data[0:1]) / (sen_data[2:3] + sen_data[0:1] + .000000001)


class NDVIComputer(VegIndexComputer):
    def compute(self, sen_data):
        return (sen_data[0:1] - sen_data[1:2]) / (sen_data[0:1] + sen_data[1:2] + .000000001)


class SentinelOpener(DataOpener):
    def __init__(self, veg_index_computers: list[VegIndexComputer] = ()):
        super().__init__('sentinel')
        self.veg_index_computers = veg_index_computers

    def open_data(self, base_path: Path):
        data = self._open_data(base_path)
        data = data.astype(np.float32) / 255
        if len(self.veg_index_computers) > 0:
            veg_indices = self.make_veg_indices(data)
            data = np.concatenate([data] + veg_indices, axis=0)
        data[:4] = data[:4] * 2 - 1
        return data

    def make_veg_indices(self, sen_data: np.ndarray):
        outputs = [computer.compute(sen_data) for computer in self.veg_index_computers]
        return outputs


class ElevationLoader(DataOpener):
    def __init__(self, elevation_name: str = 'elevation_cut',
                 include_derivatives=True, scale_value=1., rescale=True
                 ):
        super().__init__(elevation_name)
        self.include_derivatives = include_derivatives
        self.scale_value = scale_value
        self.rescale = rescale

    def open_data(self, base_path: Path):
        data = self._open_data(base_path)
        if self.rescale:
            data = self.rescale_elevation(data)
        if self.include_derivatives:
            x_der, y_der = self.make_slopes(data)
            grad = (x_der ** 2 + y_der ** 2) ** .5
            data = np.concatenate([data, x_der, y_der, grad])
        return data.astype(np.float32)

    def rescale_elevation(self, data: np.ndarray):
        data = (data - data.min()) / self.scale_value
        return data

    def make_slopes(self, data: np.ndarray):
        el_data = data[0]
        x_der = np.zeros(el_data.shape)
        y_der = np.zeros(el_data.shape)
        x_del = (el_data[:, 1:] - el_data[:, :-1])
        x_der[:, 1:-1] = (x_del[:, :-1] + x_del[:, 1:]) / 2
        x_der[:, 0] = x_del[:, 0]
        x_der[:, -1] = x_del[:, -1]

        y_del = (el_data[:-1] - el_data[1:])
        y_der[1:-1] = (y_del[:-1] + y_del[1:]) / 2
        y_der[0] = y_del[0]
        y_der[-1] = y_del[-1]

        x_der = np.stack([x_der])
        y_der = np.stack([y_der])
        return x_der, y_der


class BurnedWaterwaysOpener(DataOpener):
    def __init__(self, value_dict: dict = None):
        super().__init__('waterways_burned')
        if value_dict is None:
            value_dict = {}
        self.value_dict = value_dict

    def open_data(self, base_path):
        data = self._open_data(base_path)
        data = data.astype(np.float32)
        datac = data.copy()
        for key, value in self.value_dict.items():
            data[datac == key] = value
        return data


class Loader:
    def __init__(self, **kwargs):
        self._inputs = {key: value for key, value in kwargs.items()}

    def save(self, save_path):
        save_yaml(save_path, self._inputs)


class SenElBurnedLoader(Loader):
    def __init__(
            self, include_veg_indices: bool = True,
            elevation_name: str = 'elevation_cut',
            include_derivatives: bool = True,
            elevation_scale_value: float = 1.,
            value_dict: dict = None,
            **kwargs
    ):
        self._inputs = dict(
            include_veg_indices=include_veg_indices, elevation_name=elevation_name, value_dict=value_dict,
            include_derivatives=include_derivatives, elevation_scale_value=elevation_scale_value
        )
        self._inputs.update(kwargs)
        super().__init__(**self._inputs)
        veg_computers = []
        if include_veg_indices:
            veg_computers = self.make_veg_computers()
        self.sentinel_opener = SentinelOpener(veg_index_computers=veg_computers)
        self.elevation_opener = ElevationLoader(
            elevation_name, include_derivatives=include_derivatives,
            scale_value=elevation_scale_value
        )
        self.burned_opener = BurnedWaterwaysOpener(value_dict=value_dict)

    def open_data(self, base_path: Path) -> np.ndarray:
        sen_data = self.sentinel_opener.open_data(base_path)
        elevation_data = self.elevation_opener.open_data(base_path)
        burned_opener = self.burned_opener.open_data(base_path)
        return np.concatenate([sen_data, elevation_data, burned_opener], axis=0)

    def make_veg_computers(self) -> list[VegIndexComputer]:
        return [NDWIComputer(), NDVIComputer()]

    @classmethod
    def load(cls, load_path):
        inputs = open_yaml(load_path)
        return cls(**inputs)


class SenElBurnedLoaderEval(Loader):
    def __init__(
            self, el_base: Path,
            include_veg_indices: bool = True,
            elevation_name: str = 'elevation_cut',
            include_derivatives: bool = True,
            elevation_scale_value: float = 1.,
            value_dict: dict = None,
            **kwargs
    ):
        self._inputs = dict(
            include_veg_indices=include_veg_indices, elevation_name=elevation_name, value_dict=value_dict,
            include_derivatives=include_derivatives, elevation_scale_value=elevation_scale_value
        )
        self.el_base = el_base
        self._inputs.update(kwargs)
        super().__init__(**self._inputs)
        veg_computers = []
        if include_veg_indices:
            veg_computers = self.make_veg_computers()
        self.sentinel_opener = SentinelOpener(veg_index_computers=veg_computers)
        self.elevation_opener = ElevationLoader(
            elevation_name, include_derivatives=include_derivatives,
            scale_value=elevation_scale_value
        )
        self.burned_opener = BurnedWaterwaysOpener(value_dict=value_dict)

    def open_data(self, base_path: Path) -> np.ndarray:
        base_el = self.el_base/base_path.name
        sen_data = self.sentinel_opener.open_data(base_path)
        elevation_data = self.elevation_opener.open_data(base_el)
        burned_opener = self.burned_opener.open_data(base_path)
        return np.concatenate([sen_data, elevation_data, burned_opener], axis=0)

    def make_veg_computers(self) -> list[VegIndexComputer]:
        return [NDWIComputer(), NDVIComputer()]

    @classmethod
    def load(cls, load_path):
        inputs = open_yaml(load_path)
        return cls(**inputs)


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
    cut_data_to_match_file_list(
        data_dir=ppaths.training_data/'elevation_cut', file_paths=file_paths, base_dir_path=base_dir_path, num_proc=num_proc
    )


def make_save_next_temp_file(input_list: list,
                             data_loader: SenElBurnedLoader,
                             temp_dir: Path,
                             temp_cut_dir: Path,
                             num_proc: int=15,
                             temp_name: str='next_input.npy'
                             ):
    data = []
    cut_next_file_set(file_paths=input_list, base_dir_path=temp_cut_dir, num_proc=num_proc)
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


def get_file_shapely_box(file_dir: Path, file_name: str = 'sentinel.tif') -> shapely.Polygon:
    file_path = file_dir / file_name
    with rio.open(file_path) as rio_f:
        bounds = rio_f.bounds
    return shapely.box(*bounds)


def get_file_info_dict(file):
    bbox = get_file_shapely_box(file)
    file_dict = {
        'file': file,
        'bbox': bbox
    }
    return file_dict


class InputListGenerator:
    def __init__(self,
                 base_path: Path = ppaths.training_data / 'model_inputs',

                 ):
        self.base_path = base_path

    def make_inputs_list(
            self, use_pruned_data: bool = False, pruned_file_name: str = 'new_inputs.parquet'
    ) -> list:
        self.input_list = []
        if use_pruned_data:
            self.prune_data(pruned_file_name)
        else:
            self.make_input_info()
        return self.input_list

    def prune_data(self, pruned_file_name):
        new_inputs_file = self.base_path / pruned_file_name
        if new_inputs_file.exists():
            to_keep_df = pd.read_parquet(new_inputs_file)
            to_keep_names = to_keep_df.file_name
            self.input_list = [self.base_path/f'input_data/{file_name}' for file_name in to_keep_names]
            np.random.shuffle(self.input_list)
        else:
            raise Exception(
                f'File Does not Exist: {new_inputs_file}'
                f'Run with use_pruned_data = False'
            )

    def make_input_info(self):
        input_data_dir = self.base_path / 'input_data'
        sub_dir_list = list(input_data_dir.glob('*'))
        num = len(sub_dir_list)
        file_names = []
        for ind, sub_dir in enumerate(sub_dir_list):
            file_names.extend(sub_dir.glob('*'))
        self.input_list = file_names
        np.random.shuffle(self.input_list)


class TestListGenerator:
    def __init__(self,
                 data_loader: SenElBurnedLoader,
                 base_path: Path,
                 temp_dir: Path,
                 temp_cut_dir: Path,
                 **kwargs):
        self.data_loader = data_loader
        self.temp_dir = temp_dir
        self.temp_cut_dir = temp_cut_dir
        self.base_path = base_path


    def make_input_info(self):
        input_data_dir = self.base_path / 'test_data'
        sub_dir_list = list(input_data_dir.glob('*'))
        num = len(sub_dir_list)
        file_names = []
        for ind, sub_dir in enumerate(sub_dir_list):
            file_names.extend(sub_dir.glob('*'))
        np.random.shuffle(file_names)
        return file_names

    def make_temp_files(self, test_files):
        make_save_next_temp_file(
            test_files, data_loader=self.data_loader, temp_dir=self.temp_dir,
            temp_cut_dir=self.temp_cut_dir, temp_name='test_input.npy'
        )

    def make_test_list(self, num_test_inds: int) -> (list, list):
        test_files = self.make_input_info()
        test_files = test_files[:num_test_inds]
        self.make_temp_files(test_files)
        return test_files




class WaterwayDataLoaderV3:
    def __init__(self,
                 num_test_inds: int = 1000,
                 num_training_images_per_load: int = 2500,
                 data_loader=SenElBurnedLoader(),
                 use_pruned_data: bool = False,
                 _from_load: bool = False,
                 base_path: Path = ppaths.training_data / 'model_inputs',
                 **kwargs):
        print('in data loader')

        self.base_path = Path(base_path)
        self.temp_dir = self.base_path / 'temp'
        self.temp_cut_dir = self.base_path/'temp_cut'
        if not self.temp_dir.exists():
            self.temp_dir.mkdir()
        if not self.temp_cut_dir.exists():
            self.temp_cut_dir.mkdir()
        self.test_data = None
        self.num_training_images = num_training_images_per_load
        self.temp_process = None
        self.data_loader = data_loader
        self.test_list_generator = TestListGenerator(
            data_loader=self.data_loader, temp_cut_dir=self.temp_cut_dir,
            temp_dir=self.temp_dir, base_path=base_path
        )
        if not _from_load:
            self.clear_temp_dir()
            input_list_generator = InputListGenerator(base_path=base_path)
            print('making input list')
            self.input_list = input_list_generator.make_inputs_list(use_pruned_data)
            self.num_files = len(self.input_list)
            print('making test list')
            self.test_list = self.test_list_generator.make_test_list(num_test_inds)
            self.num_training_inds = len(self.input_list)
            self.num_test_inds = num_test_inds
            self.current_index = 0
            self.epoch = 0
            self.make_next_temp_file(num_images=num_training_images_per_load)

    def clear_temp_dir(self):
        delete_directory_contents(self.temp_dir)
        delete_directory_contents(self.temp_cut_dir)

    def make_next_temp_file(self, num_images: int = None):
        if num_images is None:
            num_images = self.num_training_images
        kwargs = {
            'input_list': self.make_next_input_list(num_images),
            'data_loader': self.data_loader,
            'temp_dir': self.temp_dir,
            'temp_cut_dir': self.temp_cut_dir,
        }
        self.temp_process = Process(target=make_save_next_temp_file, kwargs=kwargs)
        self.temp_process.start()

    def make_next_input_list(self, num_images: int = 1000):
        file_inputs = []
        while len(file_inputs) < num_images:
            inputs = self.input_list[self.current_index]
            new_index = (self.current_index + 1) % self.num_training_inds
            if new_index < self.current_index:
                self.epoch += 1
                self.shuffle_input_list()
            self.current_index = new_index
            file_inputs.append(inputs)
        return file_inputs

    def load_training_data(self, num_images=None):
        if num_images is None:
            num_images = self.num_training_images
        self.temp_process.join()
        self.temp_process.close()
        training_data = np.load(self.temp_dir / 'next_input.npy')
        self.make_next_temp_file(num_images)
        return training_data

    def load_test_data(self):
        if not (self.temp_dir/'test_input.npy').exists():
            print('making test data')
            self.clear_temp_dir()
            self.test_list_generator.make_temp_files(self.test_list)
        cut_images = np.load(self.temp_dir/'test_input.npy')
        return cut_images

    def shuffle_input_list(self):
        np.random.shuffle(self.input_list)

    def save(self, model_dir):
        info = {
            'epoch': self.epoch,
            'current_index': self.current_index,
            'num_training_images_per_load': self.num_training_images,
            'num_test_inds': self.num_test_inds,
            'base_path': str(self.base_path),
            'data_loader_module_class': [self.data_loader.__module__, self.data_loader.__class__.__name__]
        }
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        save_dir = model_dir / 'data_info'
        if not save_dir.exists():
            save_dir.mkdir()
            training_path = save_dir / 'training_data.pkl'
            testing_path = save_dir / 'testing_data.pkl'
            save_pickle(training_path, self.input_list)
            save_pickle(testing_path, self.test_list)
            self.test_list = None
        info_path = save_dir / 'training_info.yml'
        data_loader_path = save_dir / 'data_loader_info.yml'
        save_yaml(info_path, info)
        self.data_loader.save(data_loader_path)

    @classmethod
    def load(cls,
             model_number,
             base_dir=ppaths.training_data / f'model_data',
             num_training_images=None,
             epoch=None,
             current_index=None,
             data_loader=None,
             clear_temp=False):
        load_dir = base_dir / f'model_{model_number}/data_info'
        info = open_yaml(load_dir / 'training_info.yml')
        if data_loader is None:
            data_loader_module, data_loader_class = info['data_loader_module_class']
            data_loader = getattr(
                importlib.import_module(data_loader_module), data_loader_class
            )
            info['data_loader'] = data_loader.load(load_dir / 'data_loader_info.yml')
        else:
            info['data_loader'] = data_loader
        if num_training_images is not None:
            info['num_training_images_per_load'] = num_training_images
        input_list = open_pickle(load_dir / 'training_data.pkl')
        test_list = open_pickle(load_dir / 'testing_data.pkl')
        ww_data_loader = WaterwayDataLoaderV3(_from_load=True, **info)
        delete_directory_contents(ww_data_loader.temp_cut_dir)
        ww_data_loader.epoch = info['epoch'] if epoch is None else epoch
        ww_data_loader.current_index = info['current_index'] \
            if current_index is None else current_index
        ww_data_loader.input_list = input_list
        ww_data_loader.test_list = test_list
        ww_data_loader.num_test_inds = len(test_list)
        ww_data_loader.num_training_inds = len(input_list)
        if clear_temp:
            ww_data_loader.clear_temp_dir()
        ww_data_loader.load_test_data()
        ww_data_loader.make_next_temp_file(info['num_training_images_per_load'])
        return ww_data_loader
