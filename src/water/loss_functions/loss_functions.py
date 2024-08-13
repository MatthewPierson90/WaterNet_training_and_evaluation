import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class BaseLossType(nn.Module):
    def __init__(self):
        super(BaseLossType, self).__init__()
        self.loss_list_dict = {}

    def clear_lld(self):
        for item in self.loss_list_dict:
            self.loss_list_dict[item] = []


class WaterwayLossDecreaseType(BaseLossType):
    def __init__(self, num_factors):
        super().__init__()
        self.num_factors = num_factors

    def decrease_y_max(self, y, num_factors: int = None):
        if num_factors is None:
            num_factors = self.num_factors
        for i in range(num_factors):
            y = F.max_pool2d(y, 2)
        return y

    def get_within_n(self, model, target, model_val, n):
        ex, rows, cols = np.where((model == model_val) & (model != target))
        other_val = abs(model_val - 1)
        max_rows = model.shape[-2]
        max_cols = model.shape[-1]
        total = len(rows)
        within_n = 0
        for ex, row, col in zip(ex, rows, cols):
            rm = max(row - n, 0)
            rM = min(row + n + 1, max_rows)
            cm = max(col - n, 0)
            cM = min(col + n + 1, max_cols)
            if model_val in target[ex, rm:rM, cm:cM] and other_val in model[ex, rm:rM, cm:cM]:
                within_n += 1
        return within_n, total

    def calculate_numpy_statistics(self, inputs_np, targets_np, diff):
        num_one_correct = len(inputs_np[(inputs_np == 1) & (targets_np == 1)])
        num_model_one = len(inputs_np[inputs_np == 1])
        num_targets_one = len(targets_np[targets_np == 1])
        num_correct = len(diff[diff == 0])
        num_wrong = len(diff[diff != 0])
        fp_f, total_fp = self.get_within_n(inputs_np, targets_np, 1, 1)
        fn_f, total_fn = self.get_within_n(inputs_np, targets_np, 0, 1)

        accuracy = num_correct / (num_correct + num_wrong)
        precision = num_one_correct / num_model_one if num_model_one > 0 else np.nan
        recall = num_one_correct / num_targets_one if num_targets_one > 0 else np.nan
        eps = 0. if precision + recall > 0 else 1.
        f1 = 2 * precision * recall / (precision + recall + eps)

        accuracy_f = (num_correct + fp_f + fn_f) / (num_correct + num_wrong)
        precision_f = (num_one_correct + fp_f) / num_model_one if num_model_one > 0 else np.nan
        recall_f = (num_one_correct + fp_f) / (num_targets_one + fp_f - fn_f) if (num_targets_one + fp_f - fn_f) > 0 else np.nan
        eps_f = 0. if precision_f + recall_f > 0 else 1.
        f1_f = 2 * precision_f * recall_f / (precision_f + recall_f + eps_f)

        return (
            accuracy, precision, recall, f1, accuracy_f, precision_f, recall_f, f1_f, num_correct,
            num_model_one, num_targets_one, num_one_correct, num_wrong, fp_f, fn_f
        )


def tanimoto_distance(p, t):
    top = (p*t).sum()
    bottom = (p**2 + t**2 - p*t).sum()
    if bottom != 0:
        return top/bottom
    return 1


def tanimoto_loss(p, t, w):
    p = w*p
    t = w*t
    return 1 - .5*(tanimoto_distance(p, t) + tanimoto_distance(w - p, w - t))


class WaterwayLossDecTanimoto(WaterwayLossDecreaseType):
    def __init__(self, num_factors=3, tanimoto_weight=.7, **kwargs):
        super().__init__(num_factors)
        self.name = 'WaterwayLossDec'
        self.num_factors = num_factors
        self.entropy = nn.BCELoss(reduction='none')
        self.image_totals = []
        self.tanimoto_weight = tanimoto_weight
        self.loss_list_dict = {
            'BCE': [], 'tanimoto':[],'total': [], 'a_n': [], 'p_n': [], 'r_n': [],
            'f1_n': [], 'a_f': [], 'p_f': [], 'r_f': [], 'f1': []
        }


    def append_calculated_statistics(self, inputs, targets, weights):
        inputs_np = inputs.detach().to('cpu').float().numpy()
        inputs_np = inputs_np[:, 0]
        inputs_np[inputs_np < .5] = 0
        inputs_np[inputs_np >= .5] = 1
        targets_np = targets.detach().to('cpu').float().numpy()
        targets_np = targets_np[:, 0]
        diff = inputs_np - targets_np

        outputs = self.calculate_numpy_statistics(inputs_np, targets_np, diff)
        a, p, r, f1, a_f, p_f, r_f, f1_f, _, _, _, _, _, _, _ = outputs
        self.loss_list_dict['a_n'].append(a)
        self.loss_list_dict['p_n'].append(p)
        self.loss_list_dict['r_n'].append(r)
        self.loss_list_dict['f1_n'].append(f1)

        self.loss_list_dict['a_f'].append(a_f)
        self.loss_list_dict['p_f'].append(p_f)
        self.loss_list_dict['r_f'].append(r_f)
        self.loss_list_dict['f1'].append(f1_f if not np.isnan(f1_f) else 0)

    def make_weights_mult(self, weights, targets):
        f1 = self.loss_list_dict['f1'][-1]
        r = self.loss_list_dict['r_f'][-1]
        a = self.loss_list_dict['a_f'][-1]
        if (f1 == 0 or np.isnan(f1)) and a < .90:
            weights[targets == 1] *= 2.
        elif r > .98 and a < .2:
            weights[targets == 0] += .5
        return weights

    def update_image_totals(self, image_entropy):
        self.image_totals.extend(image_entropy.detach().to('cpu').float().numpy())
        if len(self.image_totals) > 20*len(image_entropy):
            self.image_totals = self.image_totals[len(image_entropy):]

    def make_image_totals_weights(self, image_entropy):
        image_totals_mean = np.array(self.image_totals).mean()
        image_totals_std = np.array(self.image_totals).std()
        top_val = image_totals_mean + 2.5*image_totals_std
        image_weights = image_entropy.clone().detach()
        image_weights[image_weights > top_val] = 0
        return image_weights

    def forward(self, inputs, targets, **kwargs):
        weights = targets.clone()
        weights[weights == 0] = 1.
        weights[weights < 1] = 0
        targets = targets.clone()
        targets[targets < 1] = 0
        targets[targets > 0] = 1
        targets = self.decrease_y_max(targets)
        weights = self.decrease_y_max(weights)
        self.append_calculated_statistics(inputs=inputs, targets=targets, weights=weights)
        weights = self.make_weights_mult(weights=weights, targets=targets)
        bce_loss = (weights * self.entropy(inputs, targets)).mean()
        weights1 = weights.clone()
        weights1[weights1 > 1] = 1
        tanimoto = tanimoto_loss(inputs[:, 0], targets[:, 0], weights1[:, 0])
        total = tanimoto*self.tanimoto_weight + bce_loss*(1-self.tanimoto_weight)
        self.loss_list_dict['BCE'].append(bce_loss.item())
        self.loss_list_dict['tanimoto'].append(tanimoto.item())
        self.loss_list_dict['total'].append(total.item())

        return total


class WaterwayLossDecForEval(WaterwayLossDecreaseType):
    def __init__(self, num_factors=3, max_target=21):
        super().__init__(num_factors)
        self.name = 'WaterwayLossDec'
        self.entropy = nn.BCELoss(reduction='none')
        self.loss_list_dict = {
            'a_n': [], 'p_n': [], 'r_n': [], 'f1_n': [], 'a_f': [], 'p_f': [], 'r_f': [], 'f1': [],
            'num_one_correct': [], 'num_model_one': [], 'num_target_one': [],
            'num_correct': [], 'num_wrong': [], 'num_fp_f': [], 'num_fn_f': []
        }
        self.summed_values = {'num_one_correct': 0, 'num_model_one': 0, 'num_target_one': 0,
            'num_correct': 0, 'num_wrong': 0, 'num_one_correct_f': 0, 'num_model_one_f': 0, 'num_target_one_f': 0,
            'num_correct_f': 0, 'num_wrong_f': 0}
        
        self.pixel_stats = {
            'a_pixel': 0., 'p_pixel': 0., 'r_pixel': 0., 'f1_pixel': 0.,
            'a_pixel_f': 0., 'p_pixel_f': 0., 'r_pixel_f': 0., 'f1_pixel_f': 0.
        }
        self.max_target = max_target + 1
        for i in range(max_target + 1):
            self.loss_list_dict[f'target_val_{i}'] = []
            self.loss_list_dict[f'target_val_{i}_correct'] = []

    def calculate_statistics(self, inputs_np, targets_np):
        inputs_np_all = inputs_np
        targets_np_all = targets_np
        diff_all = inputs_np_all - targets_np_all

        for i in range(inputs_np_all.shape[0]):
            inputs_np = inputs_np_all[i:i + 1]
            targets_np = targets_np_all[i:i + 1]
            diff = diff_all[i:i + 1]
            outputs = self.calculate_numpy_statistics(inputs_np, targets_np, diff)
            a, p, r, f1, a_f, p_f, r_f, f1_f, nc, nmo, nto, noc, nw, fp_f, fn_f = outputs
            self.loss_list_dict['a_n'].append(a)
            self.loss_list_dict['p_n'].append(p)
            self.loss_list_dict['r_n'].append(r)
            self.loss_list_dict['f1_n'].append(f1)

            self.loss_list_dict['a_f'].append(a_f)
            self.loss_list_dict['p_f'].append(p_f)
            self.loss_list_dict['r_f'].append(r_f)
            self.loss_list_dict['f1'].append(f1_f)

            self.loss_list_dict['num_one_correct'].append(noc)
            self.loss_list_dict['num_model_one'].append(nmo)
            self.loss_list_dict['num_target_one'].append(nto)
            self.loss_list_dict['num_correct'].append(nc)
            self.loss_list_dict['num_wrong'].append(nw)
            self.loss_list_dict['num_fp_f'].append(fp_f)
            self.loss_list_dict['num_fn_f'].append(fn_f)
            
            self.summed_values['num_one_correct'] += noc
            self.summed_values['num_model_one'] += nmo
            self.summed_values['num_target_one'] += nto
            self.summed_values['num_correct'] += nc
            self.summed_values['num_wrong'] += nw
            
            self.summed_values['num_one_correct_f'] += noc + fp_f
            self.summed_values['num_model_one_f'] += nmo
            self.summed_values['num_target_one_f'] += nto + fp_f - fn_f
            self.summed_values['num_correct_f'] += nc + fp_f + fn_f
            self.summed_values['num_wrong_f'] += nw - fp_f - fn_f
        self.pixel_stats['a_pixel'] = self.summed_values['num_correct'] / (
                self.summed_values['num_correct'] + self.summed_values['num_wrong']
        )
        self.pixel_stats['p_pixel'] = self.summed_values['num_one_correct'] / self.summed_values['num_model_one']
        self.pixel_stats['r_pixel'] = self.summed_values['num_one_correct'] / self.summed_values['num_target_one']
        self.pixel_stats['f1_pixel'] = 2*self.pixel_stats['p_pixel']*self.pixel_stats['r_pixel'] / (
                self.pixel_stats['p_pixel'] + self.pixel_stats['r_pixel']
        )

        self.pixel_stats['a_pixel_f'] = self.summed_values['num_correct_f']/(
                self.summed_values['num_correct_f'] + self.summed_values['num_wrong_f']
        )
        self.pixel_stats['p_pixel_f'] = self.summed_values['num_one_correct_f']/self.summed_values['num_model_one_f']
        self.pixel_stats['r_pixel_f'] = self.summed_values['num_one_correct_f']/self.summed_values['num_target_one_f']
        self.pixel_stats['f1_pixel_f'] = 2*self.pixel_stats['p_pixel_f']*self.pixel_stats['r_pixel_f']/(
                self.pixel_stats['p_pixel_f'] + self.pixel_stats['r_pixel_f']
        )
    def forward(self, inputs, targets, **kwargs):
        targets = targets.clone()
        values = targets.clone()
        targets[targets > 0] = 1
        targets = self.decrease_y_max(targets)
        values = self.decrease_y_max(values)
        values = values.to('cpu').float().numpy()
        inputs_np = inputs.detach().to('cpu').float().numpy()
        inputs_np[inputs_np < .5] = 0
        inputs_np[inputs_np >= .5] = 1
        targets_np = targets.detach().to('cpu').float().numpy()
        for target_val in range(self.max_target):
            for ind, slice in enumerate(values):
                if np.any(slice == target_val):
                    target = targets_np[ind]
                    model = inputs_np[ind]
                    self.loss_list_dict[f'target_val_{target_val}'].append(len(np.where(slice == target_val)[0]))
                    self.loss_list_dict[f'target_val_{target_val}_correct'].append(
                        len(np.where((slice == target_val) & (target == model))[0])
                    )
                else:
                    self.loss_list_dict[f'target_val_{target_val}'].append(0)
                    self.loss_list_dict[f'target_val_{target_val}_correct'].append(0)
        self.calculate_statistics(inputs_np=inputs_np[:, 0], targets_np=targets_np[:, 0])

