import torch
from water.basic_functions import ppaths
from water.models.decrease_conv10ac import WaterwayModel
from water.loss_functions.loss_functions import WaterwayTrailLossDecW1
from water.data_functions.load.load_waterway_data import (WaterwayDataLoaderV3, SenElBurnedTrailLoader)
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
    mult = 2.5
    ww_value_dict = {
        1: 0.0, # playa
        2: 0.0, # Inundation
        3: 0.5, # Swamp I
        4: 1.0, # Swamp P
        5: 1.0, # Swamp
        6: 0.5, # Reservoir
        7: 0.5, # Lake I
        8: mult*3.25, # Lake P
        9: mult*3.25, # Lake
        10: 0.0, # spillway
        11: 0.5, # drainage
        12: 1.5, # wash
        13: 0.5, # canal storm
        14: mult*1.0, # canal aqua
        15: 0.5, # canal
        16: mult*2.5, # artificial path
        17: mult*3.5, # Ephemeral
        18: mult*3.75, # Intermittent
        19: mult*3.25, # Perennial
        20: mult*3.25, # streams
        21: 1.5, # other
    }
    trail_value_dict = {
        1: mult*3.5, 2: mult*4.
    }
    container_inputs = {
        'wwm': {
            'model_class': WaterwayModel,
            'model_kwargs': dict(
                    init_channels=10, num_encoders=num_encoders, num_decoders=num_encoders-dec_steps, num_channels=16,
                    dtype=dtype, track_running_stats=False, affine=False, num_outputs=2,
                    padding_mode='zeros', device=device
            ),
            'loss_class': WaterwayTrailLossDecW1,
            'loss_kwargs': dict(num_factors=dec_steps, ww_scale_factor=2.0),
            'optimizer_class': torch.optim.SGD,
            'optimizer_kwargs': dict(lr=.0625, momentum=0.9, weight_decay=.0001),
            # 'optimizer_class': torch.optim.Adam,
            # 'optimizer_kwargs': dict(lr=0.002, betas=(0.9, 0.999)),
            'scheduler_class': BatchSizeScheduler,
            'scheduler_kwargs': dict(mode='max', step_size=1, patience=15, initial_iterations=1),
            'min_max': 'max', 'step_metric': 'f1'
        }
    }
    # model_container = ModelTrainingContainer.from_inputs(
    #         container_inputs=container_inputs, device=device,
    #         dtype=dtype, is_terminal=True, num_epochs=5
    # )
    # model_container = ModelTrainingContainer.copy_container(
    #     model_number=775, num_epochs=5, copy_optimizer=False, copy_lr_scheduler=False, copy_total_list=False
    # )
    # # model_container.model_container.schedule_dict['wwm'].current_best = 0
    # model_container.model_container.schedule_dict['wwm'].required_iterations = 5
    # model_container.model_container.schedule_dict['wwm'].step_size = 1
    # # model_container.model_container.schedule_dict['wwm'].max_iterations = 80
    # model_container.set_lr(.0625, model_name='wwm')
    # model_container = ModelTrainingContainer.load_container(model_number=767, num_epochs=5)
    # data_loader = SenElBurnedTrailLoader(
    #     el_base=ppaths.waterway/'model_inputs_224/temp_cut', elevation_name='elevation_cut',
    #     waterway_value_dict=ww_value_dict, trail_value_dict=trail_value_dict
    # )
    # data_load = WaterwayDataLoaderV3(
    #         num_training_images_per_load=1500,
    #         use_pruned_data=False, num_test_inds=2500,
    #         base_path=ppaths.waterway/'model_inputs_224',
    #         data_loader=data_loader
    # )
    # data_load = WaterwayDataLoaderV3.load(
    #     model_number=777, data_loader=data_loader, clear_temp=False, current_index=0
    # )
    # train_model(
    #         model_container=model_container, data_loader=data_load, batch_size=50,
    #         percent_data_per_inner_loop=1, augment_data=False, num_y=2
    # )
    #
    model_container = ModelTrainingContainer.copy_container(
        model_number=784, copy_optimizer=False, copy_lr_scheduler=False, num_epochs=5
    )
    # model_container = ModelTrainingContainer.load_container(model_number=782, num_epochs=4)
    # model_container.model_container.schedule_dict['wwm'].current_best = 0
    model_container.model_container.schedule_dict['wwm'].required_iterations = 5
    model_container.model_container.schedule_dict['wwm'].step_size = 1
    # model_container.model_container.schedule_dict['wwm'].max_iterations = 80
    model_container.set_lr(.01, model_name='wwm')
    data_loader = SenElBurnedTrailLoader(
        el_base=ppaths.waterway/'model_inputs_832/temp_cut', elevation_name='elevation_cut',
        waterway_value_dict=ww_value_dict, trail_value_dict=trail_value_dict
    )
    data_load = WaterwayDataLoaderV3(
        num_training_images_per_load=160,
        use_pruned_data=False,
        num_test_inds=400,
        base_path=ppaths.waterway/'model_inputs_832',
        data_loader=data_loader
    )
    # data_load = WaterwayDataLoaderV3.load(
    #     782, data_loader=data_loader, clear_temp=False, current_index=0
    # )
    train_model(
            model_container=model_container, data_loader=data_load, batch_size=4,
            percent_data_per_inner_loop=1, augment_data=False, num_y=2
    )
