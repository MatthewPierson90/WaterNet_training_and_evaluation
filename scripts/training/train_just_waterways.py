import torch
from water.basic_functions import ppaths
# from water.models.decrease_conv10ac import WaterwayModel
from waternet.model import WaterwayModel
from water.loss_functions.loss_functions import WaterwayLossDecTanimoto, WaterwayLossDecW1
from water.data_functions.load.load_waterway_data import (WaterwayDataLoaderV3, SenElBurnedLoaderEval)
from water.training.batch_scheduler import BatchSizeScheduler
from water.training.model_container import ModelTrainingContainer
from water.training.training_loop_data_increase import train_model

if __name__ == '__main__':
    load_path_dict = {}
    dec_steps = 1
    num_encoders = 5
    dtype = torch.bfloat16
    device = 'cuda'
    use_pruned_data = False
    mult = 2.0
    ww_value_dict = {
        1: 0.0, # playa
        2: 0.0, # Inundation
        3: 0.5, # Swamp I
        4: 0.5, # Swamp P
        5: 0.5, # Swamp
        6: mult*1.0, # Reservoir
        7: 0.5, # Lake I
        8: mult*3.5, # Lake P
        9: mult*3.5, # Lake
        10: 0.0, # spillway
        11: 0.5, # drainage
        12: 0.5, # wash
        13: 0.5, # canal storm
        14: 1.0, # canal aqua
        15: 0.5, # canal
        16: 1.0, # artificial path
        17: mult*3.75, # Ephemeral
        18: mult*3.75, # Intermittent
        19: mult*3.25, # Perennial
        20: mult*3.25, # streams
        21: 1.0, # other
    }
    container_inputs = {
        'wwm': {
            'model_class': WaterwayModel,
            'model_kwargs': dict(
                    init_channels=10, num_encoders=num_encoders, num_decoders=num_encoders-dec_steps, num_channels=16,
                    dtype=dtype, track_running_stats=False, affine=False, num_outputs=1,
                    padding_mode='zeros', device=device
            ),
            'loss_class': WaterwayLossDecTanimoto,
            # 'loss_class': WaterwayLossDecW1,
            'loss_kwargs': dict(num_factors=dec_steps, tanimoto_weight=0.7),
            'optimizer_class': torch.optim.SGD,
            'optimizer_kwargs': dict(lr=.0625, momentum=0.9, weight_decay=.0001),
            # 'optimizer_class': torch.optim.Adam,
            # 'optimizer_kwargs': dict(lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-5),
            'scheduler_class': BatchSizeScheduler,
            'scheduler_kwargs': dict(mode='max', step_size=1, patience=15, initial_iterations=1),
            'min_max': 'max', 'step_metric': 'f1'
        }
    }
    # model_container = ModelTrainingContainer.from_inputs(
    #         container_inputs=container_inputs, device=device,
    #         dtype=dtype, is_terminal=True, num_epochs=3
    # )
    # model_container = ModelTrainingContainer.copy_container(
    #     838, num_epochs=3, copy_optimizer=False, copy_lr_scheduler=False
    # )
    # model_container = ModelTrainingContainer.load_container(840, num_epochs=3, ckpt=7)
    # model_container.current_iteration = 8
    # model_container.model_container.schedule_dict['wwm'].required_iterations = 2
    # data_loader = SenElBurnedLoaderEval(
    #     el_base=ppaths.waterway/'model_inputs_224/temp_cut', elevation_name='elevation_cut',
    #     value_dict=ww_value_dict
    # )
    # data_load = WaterwayDataLoaderV3(
    #         num_training_images_per_load=1000,
    #         use_pruned_data=False, num_test_inds=2500,
    #         base_path=ppaths.waterway/'model_inputs_224',
    #         data_loader=data_loader
    # )
    # data_load = WaterwayDataLoaderV3.load(
    #     model_number=840, data_loader=data_loader, clear_temp=False
    # )
    # train_model(
    #         model_container=model_container, data_loader=data_load, batch_size=50, train_save_steps=1,
    #         percent_data_per_inner_loop=1, augment_data=False, num_y=1, test_every_n=8,  test_save_steps=1
    # )

    # model_number = model_container.model_number
    # del model_container
    model_container = ModelTrainingContainer.copy_container(
        model_number=840, copy_optimizer=False, copy_lr_scheduler=False, num_epochs=5
    )
    # model_container = ModelTrainingContainer.load_container(model_number=842, num_epochs=5, ckpt=7)
    model_container.model_container.schedule_dict['wwm'].current_best = 0
    model_container.model_container.schedule_dict['wwm'].required_iterations = 5
    model_container.model_container.schedule_dict['wwm'].step_size = 1
    model_container.model_container.schedule_dict['wwm'].max_iterations = 40
    model_container.set_lr(.01, model_name='wwm')
    data_loader = SenElBurnedLoaderEval(
        el_base=ppaths.waterway/'model_inputs_832/temp_cut', elevation_name='elevation_cut',
        value_dict=ww_value_dict
    )
    data_load = WaterwayDataLoaderV3(
        num_training_images_per_load=200, num_test_inds=400, use_pruned_data=False,
        base_path=ppaths.waterway/'model_inputs_832', data_loader=data_loader,
    )
    # data_load = WaterwayDataLoaderV3.load(
    #     842, data_loader=data_loader, clear_temp=False, current_index=14400
    # )
    train_model(
            model_container=model_container, data_loader=data_load, batch_size=4, train_save_steps=3,
            percent_data_per_inner_loop=1, augment_data=False, num_y=1, test_every_n=8, test_save_steps=10
    )
