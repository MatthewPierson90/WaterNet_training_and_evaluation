from water.data_functions.load.load_waterway_data import SenElBurnedLoaderEval
from water.loss_functions.loss_functions import WaterwayLossDecForEval
from water.paths import ppaths
from multiprocessing import Process
from water.training.test_model import evaluate_on_all_sen_data_multi


if __name__ == '__main__':
    model_number = 841
    max_per_it = 32

    kwargs = dict(
        model_number=model_number, evaluation_dir=ppaths.training_data/'model_inputs_832',
        num_per_load=max_per_it*2, max_per_it=max_per_it,
        data_loader=SenElBurnedLoaderEval(
            el_base=ppaths.training_data/'model_inputs_832/temp_cut', elevation_name='elevation_cut'
        ), max_target=21, input_dir_name='test_data', output_dir_name='output_test_data', stats_name='evaluation_test',
        loss_func_type=WaterwayLossDecForEval, num_y=1
    )
    p = Process(target=evaluate_on_all_sen_data_multi, kwargs=kwargs)
    p.start()
    p.join()
    p.close()

    kwargs = dict(
        model_number=model_number, evaluation_dir=ppaths.training_data/'model_inputs_832',
        num_per_load=max_per_it*2, max_per_it=max_per_it,
        data_loader=SenElBurnedLoaderEval(
            el_base=ppaths.training_data/'model_inputs_832/temp_cut', elevation_name='elevation_cut'
        ), max_target=21, input_dir_name='val_data', output_dir_name='output_val_data', stats_name='evaluation_val',
        loss_func_type=WaterwayLossDecForEval, num_y=1
    )
    p = Process(target=evaluate_on_all_sen_data_multi, kwargs=kwargs)
    p.start()
    p.join()
    p.close()

    # kwargs = dict(
    #     model_number=model_number, evaluation_dir=ppaths.training_data/'model_inputs_832',
    #     num_per_load=2*max_per_it, max_per_it=max_per_it,
    #     data_loader=SenElBurnedLoaderEval(
    #         el_base=ppaths.training_data/'model_inputs_832/temp_cut', elevation_name='elevation_cut'
    #     ), max_target=21, input_dir_name='input_data', output_dir_name='output_data', stats_name='evaluation_stats',
    #     loss_func_type=WaterwayLossDecForEval, num_y=1
    # )
    # p = Process(target=evaluate_on_all_sen_data_multi, kwargs=kwargs)
    # p.start()
    # p.join()
    # p.close()
