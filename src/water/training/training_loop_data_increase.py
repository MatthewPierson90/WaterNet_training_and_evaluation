import torch
import numpy as np
from water.data_functions.load.load_training_data import augment_4d_data_func
from water.basic_functions import ppaths, tt, time_elapsed
import shutil
from collections import defaultdict


def shuffle_data(data, weights=None):
    indices = list(range(data.shape[0]))
    np.random.shuffle(indices)
    data = data[indices]
    if weights is not None:
        weights = weights[indices]
    return data, weights


def train_on_batch(model_container: 'ModelContainer',
                   x, y, weights, to_save, save_ind, data_saver: 'DataSaver',
                   test_ind=9, model='wwm'):
    y_model = model_container.evaluate_model(
        model, input=x, train=not test_ind == save_ind
    )
    if to_save:
        data_to_save = y_model
        if type(y_model) == tuple:
            data_to_save = torch.concat(y_model, dim=1)
        data_saver.add_data(name='y_model', data=data_to_save.detach().to('cpu').float().numpy())
        # if include_gen:
    if (torch.any(torch.isnan(y_model)).detach().item()
            or torch.any(torch.isinf(y_model)).detach().item()):
        print(f'x had nan {torch.any(torch.isnan(x)).detach().item()}')
        print(f'x had inf {torch.any(torch.isinf(x)).detach().item()}')

    else:
        required_iterations = model_container.get_required_iterations(model)
        current_iteration = model_container.get_current_iteration(model)
        loss = model_container.evaluate_loss_function(
            model_name=model, inputs=y_model, targets=y,
            epoch=model_container.current_epoch,
            update_weights=(required_iterations-1 == current_iteration) and (save_ind < test_ind)
        )
        if save_ind < test_ind:
            num_iterations = model_container.get_required_iterations(model)
            loss = loss/num_iterations
            loss.backward(retain_graph=False)
            model_container.update_model(model)


def make_tensors(data, weights, device, dtype, num_y=1):
    x = torch.tensor(data[:, :-num_y], dtype=dtype, device=device)
    y = torch.tensor(data[:, -num_y:], dtype=dtype, device=device)
    # weights = torch.tensor(weights, dtype=dtype, device=device)
    weights = None
    return x, y, weights


def make_weights(data):
    weights = np.ones((data.shape[0], 2), dtype=np.int8)
    weights[:, 0] = 1
    return weights



def test_model(model_container: 'ModelContainer',
               test_data,
               test_weights,
               batch_size,
               epoch,
               num_its,
               max_data_index,
               data_index,
               num_y=1,
               save_steps=2
               ):
    device = model_container.device
    dtype = model_container.dtype
    save_ind = 9
    len_data = (len(test_data) // batch_size) * batch_size
    num_steps = len(list(range(0, len_data, batch_size)))
    test_data, test_weights = shuffle_data(test_data, test_weights)
    data_saver = DataSaver(num_steps=save_steps, save_ind=9)
    to_save_inds = np.random.choice(range(0, num_steps), size=save_steps, replace=False)
    if test_weights is None:
        test_weights = make_weights(test_data)
    with torch.no_grad():
        for ind, batch_step in enumerate(range(0, len_data, batch_size)):
            x, y, weights = make_tensors(
                data=test_data[batch_step: batch_step + batch_size], dtype=dtype, num_y=num_y,
                weights=test_weights[batch_step: batch_step + batch_size], device=device
            )
            to_save = ind in to_save_inds
            if to_save:
                data_saver.add_data(name='raw', data=test_data[batch_step: batch_step + batch_size])
            train_on_batch(
                model_container=model_container, x=x, y=y, weights=weights,
                to_save=to_save, save_ind=save_ind, data_saver=data_saver
            )
            del x
            del y
            if ind % 5 == 0:
                model_container.print_all_information(
                    batch_step=batch_step, num_its=1, count=save_ind, data_index=data_index,
                    save_ind=save_ind, max_data_index=max_data_index, len_data=len_data
                )
        data_saver.save_data()

        model_container.print_all_information(
            batch_step=len_data, num_its=1, count=save_ind, data_index=data_index,
            save_ind=save_ind, max_data_index=max_data_index, len_data=len_data
        )
        model_container.update_all_total_dicts_and_schedulers(exclude_scheduler=False, epoch=epoch)
        model_container.update_and_save_training_log(
            epoch=epoch, data_index=data_index, save_ind=save_ind
        )
        model_container.clear_all_model_loss_list_dicts()


def train_inner_loop(model_container: 'ModelContainer',
                     training_data,
                     training_weights,
                     batch_size,
                     num_data,
                     data_per,
                     current_iteration,
                     epoch,
                     num_its,
                     data_index,
                     max_data_index,
                     num_y=1,
                     augment_data=True,
                     save_steps=2
                     ):
    device = model_container.device
    dtype = model_container.dtype
    s = tt()
    training_data, training_weights = shuffle_data(training_data, training_weights)
    time_elapsed(s, 2)
    num_data_batches = num_data//data_per
    data_saver = DataSaver(num_steps=save_steps, save_ind=current_iteration % 8 + 1)
    for count, inner_epoch in enumerate(range(0, num_data, data_per)):
        training_data_sub = training_data[inner_epoch:inner_epoch + data_per].copy()
        if augment_data:
            s = tt()
            training_data_sub = augment_4d_data_func(training_data_sub)
            time_elapsed(s, 2)
        np.random.shuffle(training_data_sub)
        len_data = (len(training_data_sub) // batch_size) * batch_size
        num_its = len(range(0, len_data, batch_size))
        x, y, weights = make_tensors(
            data=training_data_sub, dtype=dtype, num_y=num_y,
            weights=None, device=device
        )
        to_save_inds = np.random.choice(range(0, num_its - 1), size=save_steps, replace=False)

        for ind, batch_step in enumerate(range(0, len_data, batch_size)):
            x_sub, y_sub = x[batch_step: batch_step + batch_size], y[batch_step: batch_step + batch_size]
            to_save = ind in to_save_inds
            if to_save:
                training_slice = training_data_sub[batch_step: batch_step + batch_size]
                data_saver.add_data(name='raw', data=training_slice.astype(np.float32))
            train_on_batch(
                model_container=model_container, x=x_sub, y=y_sub, weights=weights,
                to_save=to_save, save_ind=current_iteration % 8 + 1, data_saver=data_saver
            )
            del x_sub, y_sub
            if ind % 5 == 0:
                model_container.print_all_information(
                    batch_step=batch_step, num_its=num_data_batches, count=count,
                    data_index=data_index, save_ind=current_iteration % 8 + 1,
                    max_data_index=max_data_index, len_data=len_data
                )
        del x, y
        model_container.print_all_information(
            batch_step=len_data, num_its=num_data_batches, count=count,
            data_index=data_index, save_ind=current_iteration%8 + 1,
            max_data_index=max_data_index, len_data=len_data
        )
        current_iteration += 1
        data_saver.save_data()
        model_container.save_all(save_ind=current_iteration % 8 + 1, epoch=epoch)

        model_container.update_and_save_training_log(
            epoch=epoch, data_index=data_index, save_ind=count
        )
        model_container.clear_all_model_loss_list_dicts()
    return current_iteration


class DataSaver:
    def __init__(self, num_steps=2, save_ind=0):
        self.save_path = ppaths.training_data / f'model_results/training_batch_{save_ind}'
        self.data_to_save = defaultdict(list)
        self.num_steps = num_steps
        self.current_step = 0

    def add_data(self, name, data):
        self.data_to_save[name].append(data)

    def save_data(self):
        shutil.rmtree(self.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        for name, data_list in self.data_to_save.items():
            np.save(self.save_path/name, np.concatenate(data_list, axis=0))



def train_model(model_container: 'ModelTrainingContainer',
                data_loader,
                batch_size=15,
                percent_data_per_inner_loop=1/8,
                min_images_per_load=500,
                max_data_index=None,
                augment_data=True,
                num_y=1,
                test_every_n=8,
                train_save_steps=1,
                test_save_steps=1,
                ):
    start_epoch = model_container.current_epoch
    num_epochs = model_container.num_epochs
    count = 0
    if max_data_index is None:
        max_data_index = data_loader.num_training_inds
    else:
        data_loader.num_training_inds = max_data_index
    model_container.data_index = data_loader.current_index
    print(model_container.model_container.optimizer_dict['wwm'])
    iteration = model_container.current_iteration
    test_count = model_container.current_iteration
    print("Starting Training Loop...")
    count = 0
    for epoch in range(start_epoch, num_epochs):
        model_container.current_epoch = epoch
        while data_loader.epoch <= epoch:
            print('Loading training data')
            training_data = data_loader.load_training_data()
            st = tt()
            s = tt()
            training_weights = None
            data_loader.save(model_container.model_path)
            data_per = int(len(training_data) * percent_data_per_inner_loop)
            num_data = len(training_data)
            num_its = num_data//data_per
            time_elapsed(s, 2)
            iteration = train_inner_loop(
                model_container, training_data=training_data,
                training_weights=training_weights, batch_size=batch_size, num_data=num_data,
                data_per=data_per, epoch=epoch, num_its=num_its, save_steps=train_save_steps,
                current_iteration=iteration, data_index=model_container.data_index,
                max_data_index=max_data_index, augment_data=augment_data, num_y=num_y,
            )
            count += 1
            del training_data
            if model_container.test_model and (count % test_every_n == 0):
                model_container.test_model = False
                test_data = data_loader.load_test_data()
                test_weights = None
                test_model(
                    model_container=model_container, data_index=10000000 + test_count, save_steps=test_save_steps,
                    test_data=test_data, test_weights=test_weights, batch_size=batch_size,
                    num_its=1, epoch=epoch, max_data_index=max_data_index, num_y=num_y,
                )
                del test_data
                test_count += 1
            time_elapsed(st)
            model_container.data_index = data_loader.current_index

