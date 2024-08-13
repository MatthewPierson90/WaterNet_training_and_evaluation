import torch
from water.basic_functions import save_yaml, open_yaml
from water.paths import ppaths
from water.training.print_info import TrainingInfoPrinter
from water.loss_functions.loss_functions import BaseLossType
import pandas as pd
import os
import numpy as np
from pathlib import Path
import importlib


"""
This project was originally birthed from a separate more general computer vision project. In the more general project,
GAN models were being used, and it was desirable to have a single object that could deal with multiple models. Since
we are only ever using a single model in this project, the model container is way more complex than it needs to be,
but refactoring the code to only deal with a single model is unnecessary at this point.

Further, the original goal was to have a model container for training and a model container for evaluation, but since
the training container does everything that an evaluation container would need to do, and doesn't require much more in
terms of memory, I simply use the training container everywhere.
"""


def get_most_recent_epoch_checkpoint_path(
        model_name: str,
        base_path: Path = ppaths.training_data / 'model_data',
        model_number: int = None,
        epoch: int = None
):
    """
    Get the path of the most recent checkpoint for a given model and epoch.

    Args:
        model_name (str): The name of the model.
        base_path (Path): The base directory where model checkpoints are stored.
        model_number (int): The model number (default is None).
        epoch (int): The epoch number (default is None).

    Returns:
        Tuple[Path, str]: A tuple containing the path to the most recent checkpoint file and the checkpoint number as a string.
    """
    if model_number is None:
        model_number = get_most_recent_model_number(base_path)
    if epoch is None:
        epoch = get_most_recent_epoch(
            model_number=model_number, model_name=model_name, base_path=base_path
        )
    epoch_path = base_path / f'model_{model_number}/{model_name}/epoch_{epoch}'
    checkpoints = epoch_path.glob('*')
    most_recent_file = None
    checkpoint_num = None
    recent_time = 0

    for file in checkpoints:
        mtime = file.stat().st_mtime
        if mtime > recent_time:
            recent_time = mtime
            most_recent_file = file
            checkpoint_num = most_recent_file.name.split('_')[-1].split('.ckpt')[0]
    return most_recent_file, checkpoint_num


def get_most_recent_model_number(data_dir: Path = ppaths.training_data / 'model_data'):
    """
    Get the most recent model number from a directory.

    Args:
        data_dir (Path): The directory where model data is stored.

    Returns:
        int: The most recent model number.
    """
    files = data_dir.glob('model_*')
    largest = 0
    for file in files:
        model_num = int(file.name.split('model_')[-1])
        if model_num > largest:
            largest = model_num
    return largest


def get_most_recent_epoch(model_number, model_name: str = 'wwm',
                          base_path: Path = ppaths.training_data / f'model_data'
                          ):
    """
    Get the most recent epoch for a specific model.

    Args:
        model_number (int): The model number.
        model_name (str): The name of the model (default is 'wwm').
        base_path (Path): The base directory where model data is stored.

    Returns:
        int: The most recent epoch number.
    """
    data_dir = base_path / f'model_{model_number}/{model_name}'
    files = data_dir.glob('epoch_*')
    largest = 0
    for file in files:
        model_num = int(file.name.split('epoch_')[-1])
        if model_num > largest:
            largest = model_num
    return largest


class TrainingLogger:
    # Class for logging training information
    def __init__(self, model_path: Path):
        super().__init__()
        self.training_log_path = model_path / 'training_logs'
        if not self.training_log_path.exists():
            self.training_log_path.mkdir()

    def update_and_save_training_log(self, epoch, data_index, save_ind, loss_dict: 'LossDict'):
        log_name = f'epoch_{epoch}_data_index_{data_index}_waterways.csv'
        save_path = self.training_log_path / log_name
        df = pd.DataFrame()
        for model in loss_dict:
            lld = loss_dict.get_model_loss_list_dict(model)
            for error_type in lld:
                if len(lld[error_type]) > 0:
                    df[f'{model}_{error_type}'] = lld[error_type]
        df['epoch'] = epoch
        df['data_index'] = data_index
        df['save_index'] = save_ind
        if save_path.exists():
            current_df = pd.read_csv(save_path)
            df = pd.concat([current_df, df], ignore_index=True)
        df.to_csv(save_path, index=False)


class ModelLogs(dict):
    def add_model_to_model_log(self,
                               model_name: str,
                               model_class: torch.nn.Module,
                               model_kwargs: dict,
                               optimizer_class,
                               optimizer_kwargs: dict,
                               scheduler_class,
                               scheduler_kwargs: dict,
                               loss_class: BaseLossType,
                               loss_kwargs: dict,
                               min_max: str,
                               step_metric: str
                               ):
        model_kwargs_for_save = {key: value for key, value in model_kwargs.items()
                                 if key not in ['device', 'dtype']}
        if 'dtype' in model_kwargs:
            model_kwargs_for_save['dtype'] = str(model_kwargs['dtype']).split('torch.')[-1]
        self[model_name] = {
            'model_class': [model_class.__module__, model_class.__name__],
            'model_kwargs': model_kwargs_for_save,
            'optimizer_class': [optimizer_class.__module__, optimizer_class.__name__],
            'optimizer_kwargs': optimizer_kwargs,
            'scheduler_class': [scheduler_class.__module__, scheduler_class.__name__],
            'scheduler_kwargs': {key: value for key, value in scheduler_kwargs.items() if 'optimizer' != key},
            'loss_class': [loss_class.__module__, loss_class.__name__],
            'loss_kwargs': loss_kwargs,
            'min_max': min_max,
            'step_metric': step_metric
        }

    def save(self, model_path):
        save_yaml(model_path / 'model_logs.yaml', dict(self))


class ModelDict(dict[str, torch.nn.Module]):
    def evaluate_model(self, model_name, input, train=True, **kwargs):
        model = self[model_name]
        if train == True:
            model.train()
        else:
            model.eval()
        model_output = model(input, **kwargs)
        return model_output

    def zero_grad(self, model_name):
        self[model_name].zero_grad(set_to_none=True)

    def zero_all_grad(self):
        for model_name in self:
            self.zero_grad(model_name)


class LossDict(dict[str, BaseLossType]):
    def clear_model_loss_list_dict(self, model_name):
        self[model_name].clear_lld()

    def get_model_loss_list_dict(self, model_name):
        loss_function = self[model_name]
        return loss_function.loss_list_dict

    def clear_all_model_loss_list_dicts(self):
        for model_name in self:
            self.clear_model_loss_list_dict(model_name)

    def evaluate_loss_function(self, model_name, **kwargs):
        loss_function = self[model_name]
        return loss_function(**kwargs)


class OptimizerDict(dict):
    def update_model(self, model_name, model_dict: ModelDict):
        model = model_dict[model_name]
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        self[model_name].step()

    def update_all_models(self, model_dict: ModelDict, loss_dict: LossDict):
        for model_name in self:
            model_loss_list = loss_dict.get_model_loss_list_dict(model_name)
            if len(model_loss_list['total']) > 0:
                self.update_model(model_name, model_dict)



class ScheduleDict(dict):
    def __init__(self):
        super().__init__()
        self.step_metrics = {}

    def update_schedule(self, model_name, value):
        self[model_name].step(value)

    def get_current_iteration(self, model_name):
        schedule = self[model_name]
        if hasattr(schedule, 'current_iteration'):
            return self[model_name].current_iteration
        return 0
    
    def get_required_iterations(self, model_name):
        schedule = self[model_name]
        if hasattr(schedule, 'required_iterations'):
            return schedule.required_iterations
        return 1


class ModelContainer:
    def __init__(self, model_number: int, base_path: Path = ppaths.training_data / 'model_data'):
        self.model_number = model_number
        if not base_path.exists():
            base_path.mkdir()
        self.model_path = base_path / f'model_{model_number}'
        self.model_dict = ModelDict()
        self.optimizer_dict = OptimizerDict()
        self.schedule_dict = ScheduleDict()
        self.loss_dict = LossDict()
        self.printer_dict = {}
        self.total_dict = {}
        self.model_logs = ModelLogs()
        self.model_names = set()

    def update_total_dict_and_scheduler(self, model_name, exclude_scheduler=True,
                                        only_scheduler=False, epoch=0):
        total_list = self.total_dict[model_name]
        lld = self.loss_dict[model_name].loss_list_dict
        metric = self.schedule_dict.step_metrics[model_name]
        if len(lld[metric]) > 0:
            new_total = np.array(lld[metric]).mean()
            if not only_scheduler:
                total_list.append(new_total)
            if not exclude_scheduler:
                self.schedule_dict.update_schedule(model_name=model_name, value=new_total)
                to_save = False
                match self.schedule_dict[model_name].mode:
                    case 'min':
                        to_save = new_total == min(total_list)
                    case 'max':
                        to_save = new_total == max(total_list)
                if to_save:
                    self.save_model(model_name=model_name, save_ind=100, epoch=epoch)

    def update_all_total_dicts_and_schedulers(self, exclude_scheduler=True, only_scheduler=False, epoch=0):
        if exclude_scheduler:
            only_scheduler = False
        for model_name in self.model_names:
            self.update_total_dict_and_scheduler(
                model_name, exclude_scheduler=exclude_scheduler, only_scheduler=only_scheduler, epoch=epoch
            )

    def save_model(self, model_name, save_ind, epoch):
        model_save_dir = self.model_path / f'{model_name}/epoch_{epoch}'
        if not model_save_dir.exists():
            model_save_dir.mkdir(parents=True)
        save_path = model_save_dir / f'{model_name}_{save_ind}.ckpt'
        torch_save_dict_generator = {
            'epoch': epoch,
            'model_state_dict': self.model_dict[model_name].state_dict(),
            'optimizer_state_dict': self.optimizer_dict[model_name].state_dict(),
            'lr_state_dict': self.schedule_dict[model_name].state_dict(),
            'total_list': self.total_dict[model_name]
        }
        torch.save(torch_save_dict_generator, save_path)

    def save_all(self, save_ind, epoch):
        for model_name in self.model_names:
            self.save_model(model_name, save_ind=save_ind, epoch=epoch)


class ModelContainerFactory:
    def __init__(self, base_path: Path = ppaths.training_data / 'model_data'):
        self.base_path = base_path

    def _build_container(self, model_container: ModelContainer, model_name: str,
                model_class, model_kwargs: dict,
                loss_class, loss_kwargs: dict,
                optimizer_class=None, optimizer_kwargs: dict = None,
                scheduler_class=None, scheduler_kwargs: dict = None,
                min_max='max', step_metric='f1',
                ):
        optimizer_class, optimizer_kwargs = self.get_optimizer_class_kwargs(
            optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs
        )
        scheduler_class, scheduler_kwargs = self.get_scheduler_class_kwargs(
            scheduler_class=scheduler_class, scheduler_kwargs=scheduler_kwargs, min_max=min_max
        )
        model = model_class(**model_kwargs)
        model_container.model_dict[model_name] = model
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        model_container.optimizer_dict[model_name] = optimizer

        scheduler_kwargs['optimizer'] = optimizer
        model_container.schedule_dict[model_name] = scheduler_class(**scheduler_kwargs)
        model_container.schedule_dict.step_metrics[model_name] = step_metric

        loss = loss_class(**loss_kwargs)
        model_container.loss_dict[model_name] = loss

        model_container.model_names.add(model_name)
        model_container.total_dict[model_name] = []
        model_container.printer_dict[model_name] = TrainingInfoPrinter(
            min_max=min_max, loss_list_dict=loss.loss_list_dict
        )
        model_container.model_logs.add_model_to_model_log(
            model_name=model_name, model_class=model_class, model_kwargs=model_kwargs,
            loss_class=loss_class, loss_kwargs=loss_kwargs, optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs, scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs, min_max=min_max, step_metric=step_metric
        )
        return model_container

    @staticmethod
    def get_optimizer_class_kwargs(optimizer_class, optimizer_kwargs):
        if optimizer_class is None:
            optimizer_class = torch.optim.Adam
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': .0005, 'betas': (.99, 0.999)}
        return optimizer_class, optimizer_kwargs

    @staticmethod
    def get_scheduler_class_kwargs(scheduler_class, scheduler_kwargs, min_max):
        if scheduler_class is None:
            scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
        if scheduler_kwargs is None:
            scheduler_kwargs = {'mode': min_max, 'factor': .8, 'patience': 25, 'min_lr': 0}
        return scheduler_class, scheduler_kwargs

    def build_container(self, inputs,
                        model_number: int = None,
                        ):
        if model_number is None:
            model_number = get_most_recent_model_number(self.base_path) + 1
        model_container = ModelContainer(model_number=model_number, base_path=self.base_path)
        for model_name, input_dict in inputs.items():
            model_container = self._build_container(
                model_name=model_name, model_container=model_container, **input_dict
            )
        model_path = self.base_path / f'model_{model_number}'
        if not model_path.exists():
            model_path.mkdir()
            model_container.model_logs.save(model_path)

        return model_container

    def load_container(self, model_number: int, epoch=None, ckpt=None):
        inputs = self._get_container_model_logs(model_number)
        model_container = self.build_container(inputs, model_number=model_number)
        model_container = self._load_weights(
            model_number=model_number, model_container=model_container, copy_lr_scheduler=True,
            copy_total_list=True, epoch=epoch, ckpt=ckpt
        )
        return model_container

    def copy_container(self, model_number: int,
                       models_to_copy: list[str] = None,
                       copy_optimizer: bool = True,
                       copy_lr_scheduler: bool = False,
                       copy_total_list: bool = False,
                       epoch=None, ckpt=None,
                       ):
        inputs = self._get_container_model_logs(model_number)
        model_container = self.build_container(inputs)
        model_container = self._load_weights(
            model_number=model_number, model_container=model_container, epoch=epoch,
            models_to_copy=models_to_copy, copy_optimizer=copy_optimizer, ckpt=ckpt,
            copy_lr_scheduler=copy_lr_scheduler, copy_total_list=copy_total_list,
        )
        return model_container

    def _load_state_dicts(self, model_number, model_name, epoch=None, ckpt=None):
        if epoch is None:
            epoch = get_most_recent_epoch(
                model_number=model_number, model_name=model_name, base_path=self.base_path
            )
        if ckpt is None:
            _, ckpt = get_most_recent_epoch_checkpoint_path(
                model_name=model_name, model_number=model_number,
                base_path=self.base_path, epoch=epoch
            )
        epoch_dir = self.base_path / f'model_{model_number}/{model_name}/epoch_{epoch}'
        print(epoch, ckpt)
        load_path = epoch_dir / f'{model_name}_{ckpt}.ckpt'
        state_dicts = torch.load(load_path)
        return state_dicts

    def _load_weights(self, model_number: int,
                      model_container: ModelContainer,
                      models_to_copy: list[str] = None,
                      copy_optimizer: bool = True,
                      copy_lr_scheduler: bool = False,
                      copy_total_list: bool = False,
                      epoch=None, ckpt=None
                      ):
        if models_to_copy is None:
            models_to_copy = model_container.model_names
        for model_to_copy in models_to_copy:
            state_dicts = self._load_state_dicts(
                model_number=model_number, model_name=model_to_copy, epoch=epoch, ckpt=ckpt
            )
            model = model_container.model_dict[model_to_copy]
            model.load_state_dict(state_dicts['model_state_dict'], strict=False)
            if copy_optimizer:
                optimizer = model_container.optimizer_dict[model_to_copy]
                optimizer.load_state_dict(state_dicts['optimizer_state_dict'])
            if copy_lr_scheduler:
                scheduler = model_container.schedule_dict[model_to_copy]
                scheduler.load_state_dict(state_dicts['lr_state_dict'])
            if copy_total_list:
                model_container.total_dict[model_to_copy] = state_dicts['total_list']
        return model_container

    def _get_container_model_logs(self, model_number: int):
        log_path = self.base_path / f'model_{model_number}/model_logs.yaml'
        model_logs = open_yaml(log_path)
        for model_inputs in model_logs.values():
            for input_name, input_info in model_inputs.items():
                if 'class' in input_name:
                    module, name = input_info
                    model_inputs[input_name] = getattr(
                        importlib.import_module(module), name
                    )
                if input_name == 'model_kwargs':
                    if 'dtype' in input_info:
                        input_info['dtype'] = getattr(torch, input_info['dtype'])
        return model_logs


class ModelTrainingContainer:
    def __init__(self,
                 device='cuda',
                 dtype=torch.bfloat16,
                 num_epochs=50,
                 is_terminal=True,
                 model_container: ModelContainer = None,
                 **kwargs,
                 ):
        self.is_terminal = is_terminal
        self.model_container = model_container
        self.model_path = model_container.model_path
        self.model_number = model_container.model_number
        self.training_logger = TrainingLogger(model_container.model_path)
        self.device = device
        self.dtype = dtype
        self.current_epoch = 0
        self.data_index = 0
        self.test_model = True
        self.num_epochs = num_epochs
        self.current_iteration = len(list(model_container.total_dict.values())[0])

    def evaluate_model(self, model_name, input, train=True, **kwargs):
        return self.model_container.model_dict.evaluate_model(
            model_name=model_name, input=input, train=train, **kwargs
        )

    def evaluate_loss_function(self, model_name, **kwargs):
        return self.model_container.loss_dict.evaluate_loss_function(model_name, **kwargs)

    def update_model(self, model_name):
        if self.get_current_iteration(model_name) == 0:
            self.model_container.optimizer_dict.update_model(
                model_name=model_name, model_dict=self.model_container.model_dict
            )
            self.model_container.model_dict.zero_grad(model_name)
            self.test_model = True
        self.model_container.schedule_dict[model_name].next_iteration()


    def update_all_models(self):
        self.model_container.optimizer_dict.update_all_models(
            model_dict=self.model_container.model_dict,
            loss_dict=self.model_container.loss_dict
        )

    def clear_all_model_loss_list_dicts(self):
        self.model_container.loss_dict.clear_all_model_loss_list_dicts()

    def update_and_save_training_log(self, epoch, data_index, save_ind):
        self.training_logger.update_and_save_training_log(
            epoch=epoch, data_index=data_index, save_ind=save_ind,
            loss_dict=self.model_container.loss_dict
        )

    def update_all_total_dicts_and_schedulers(self, exclude_scheduler=True, only_scheduler=False, epoch=0):
        self.model_container.update_all_total_dicts_and_schedulers(
            exclude_scheduler=exclude_scheduler, only_scheduler=only_scheduler, epoch=epoch
        )

    def zero_all_grad(self):
        self.model_container.model_dict.zero_all_grad()

    def print_all_information(
            self, batch_step, save_ind, num_its, count,
            data_index, max_data_index, len_data,
    ):
        if self.is_terminal:
            os.system('clear')
        print('')
        print(f'Model number: {self.model_number}')
        print(f'Current epoch: {self.current_epoch}/{self.num_epochs}')
        print(f'Dataset batch: {count + 1}/{num_its} (save num: {save_ind})')
        print('')
        loss_dict = self.model_container.loss_dict
        total_dict = self.model_container.total_dict
        printer_dict = self.model_container.printer_dict
        for model_name, loss in loss_dict.items():
            if len(loss.loss_list_dict['total']) > 0:
                model_loss_dict = {
                    model_name: {
                        'loss': loss_dict[model_name], 'total_list': total_dict[model_name]
                    }
                }
                printer_dict[model_name].print_info(
                    i=batch_step, data_index=data_index, max_data_index=max_data_index,
                    len_data=len_data, model_loss_dict=model_loss_dict
                )
                print('')
                schedule = self.model_container.schedule_dict[model_name]
                num_bad_epochs = schedule.num_bad_epochs
                patience = schedule.patience
                required_iterations = schedule.required_iterations
                patience_size = f'{len(str(patience))}d'
                print(f'Required iterations: {required_iterations}, Step count: {num_bad_epochs:{patience_size}}/{patience}')
                print('')

    def get_lr(self, model_name: str):
        return self.model_container.optimizer_dict[model_name].param_groups[0]['lr']

    def set_lr(self, new_lr: float, model_name: str):
        self.model_container.optimizer_dict[model_name].param_groups[0]['lr'] = new_lr

    def save_all(self, save_ind, epoch):
        self.model_container.save_all(save_ind=save_ind, epoch=epoch)

    def get_required_iterations(self, model_name):
        return self.model_container.schedule_dict.get_required_iterations(model_name)

    def get_current_iteration(self, model_name):
        return self.model_container.schedule_dict.get_current_iteration(model_name)

    @classmethod
    def from_inputs(cls, container_inputs: dict,
                    device='cuda',
                    dtype=torch.bfloat16,
                    num_epochs=50,
                    is_terminal=True,
                    base_path: Path = ppaths.training_data / 'model_data',
                    **kwargs):
        factory = ModelContainerFactory(base_path=base_path)
        print('making container')
        model_container = factory.build_container(container_inputs)
        # dtype = model_container.model_logs.values()
        print('making training container')

        model_trainer = ModelTrainingContainer(
            device=device, dtype=dtype, is_terminal=is_terminal,
            model_container=model_container, num_epochs=num_epochs
        )
        return model_trainer

    @classmethod
    def load_container(cls, model_number: int, epoch=None,
                       ckpt=None, base_path: Path = ppaths.training_data / 'model_data',
                       device='cuda', dtype=torch.bfloat16, is_terminal=True, num_epochs=1,
                       ):
        factory = ModelContainerFactory(base_path=base_path)
        model_container = factory.load_container(
            model_number=model_number, epoch=epoch, ckpt=ckpt
        )
        model_trainer = ModelTrainingContainer(
            device=device, dtype=dtype, is_terminal=is_terminal,
            model_container=model_container, num_epochs=num_epochs
        )
        return model_trainer

    @classmethod
    def copy_container(cls, model_number: int,
                       models_to_copy: list[str] = None,
                       copy_optimizer: bool = True,
                       copy_lr_scheduler: bool = False,
                       copy_total_list: bool = False,
                       epoch=None, ckpt=None,
                       base_path: Path = ppaths.training_data / 'model_data',
                       device='cuda', dtype=torch.bfloat16, is_terminal=True,
                       num_epochs=1,
                       ):
        factory = ModelContainerFactory(base_path=base_path)
        model_container = factory.copy_container(
            model_number=model_number, epoch=epoch, ckpt=ckpt,
            copy_optimizer=copy_optimizer, copy_lr_scheduler=copy_lr_scheduler,
            copy_total_list=copy_total_list, models_to_copy=models_to_copy
        )
        model_trainer = ModelTrainingContainer(
            device=device, dtype=dtype, is_terminal=is_terminal,
            model_container=model_container, num_epochs=num_epochs
        )
        return model_trainer

