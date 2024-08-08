import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import functional as MF
import numpy as np


class MyLossType(nn.Module):
    def __init__(self):
        super(MyLossType, self).__init__()
        self.loss_list_dict = {}

    def clear_lld(self):
        for item in self.loss_list_dict:
            self.loss_list_dict[item] = []


class WaterwayLossDecreaseType(MyLossType):
    def __init__(self, num_factors):
        super().__init__()
        self.num_factors = num_factors

    def decrease_y_max(self, y, num_factors: int = None):
        if num_factors is None:
            num_factors = self.num_factors
        for i in range(num_factors):
            y = F.max_pool2d(y, 2)
        return y

    def decrease_y_avg(self, y, num_factors: int = None):
        if num_factors is None:
            num_factors = self.num_factors
        for i in range(num_factors):
            y = F.avg_pool2d(y, 2)
        return y

    def decrease_max_average(self, y):
        y = self.decrease_y_max(y, 1)
        y = F.interpolate(y, scale_factor=2, mode='bicubic', align_corners=True)
        y = self.decrease_y_max(y, 2)
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

class WaterwayLoss(MyLossType):
    def __init__(self):
        super().__init__()
        self.name = 'WaterwayLoss'
        self.entropy = nn.BCELoss(reduction='none')
        self.loss_list_dict = {
            'BCE': [], 'total': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }

    def forward(self, inputs, targets, scale_amount=.2):
        targets = targets.clone()
        scale = targets.clone()
        scale = scale + scale_amount

        targets[targets <= .5] = 0
        targets[.5 < targets] = 1
        entropy = (scale * self.entropy(inputs, targets)).sum(dim=(1, 2, 3)).mean()

        total = entropy
        # print(entropy.mean())
        self.loss_list_dict['BCE'].append(entropy.item())
        self.loss_list_dict['total'].append(total.item())
        inputs_np = inputs.detach().to('cpu').float().numpy()
        inputs_np[inputs_np < .5] = 0
        inputs_np[inputs_np >= .5] = 1

        targets_np = targets.detach().to('cpu').float().numpy()
        targets_np[targets_np < .5] = 0
        targets_np[targets_np >= .5] = 1
        num_zero = len(targets_np[targets_np == 0])
        if num_zero == 0:
            num_zero = .001
        num_one = len(targets_np[targets_np == 1])
        if num_one == 0:
            num_one = .001
        num_one_inp = len(inputs_np[inputs_np == 1])
        if num_one_inp == 0:
            num_one_inp = .001
        num_total = num_one + num_zero

        # num_zero_correct = len(inputs_np[(targets_np == 0) & (inputs_np == 0)])
        num_one_correct = len(inputs_np[(targets_np == 1) & (inputs_np == 1)])
        # num_one = len(np.)
        num_total_correct = len(inputs_np[(targets_np == inputs_np)])
        precision = num_one_correct / num_one_inp
        recall = num_one_correct / num_one
        eps = 0. if precision + recall > 0 else 1.
        f1 = 2 * precision * recall / (precision + recall + eps)
        self.loss_list_dict['accuracy'].append(num_total_correct / num_total)
        self.loss_list_dict['precision'].append(precision)
        self.loss_list_dict['recall'].append(recall)
        self.loss_list_dict['f1'].append(f1)
        return total



class WaterwayLossDecW(WaterwayLossDecreaseType):
    def __init__(self, num_factors=3):
        super().__init__(num_factors)
        self.name = 'WaterwayLossDec'
        self.num_factors = num_factors
        self.entropy = nn.BCELoss(reduction='none')
        self.loss_list_dict = {
            'BCE': [], 'total': [], 'a_n': [], 'p_n': [], 'r_n': [],
            'f1_n': [], 'a_f': [], 'p_f': [], 'r_f': [], 'f1': []
        }

    def append_calculated_statistics(self, inputs, targets):

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
        self.loss_list_dict['f1'].append(f1_f)

    def make_weights_mult(self, weights, targets, inputs):
        f1_list = self.loss_list_dict['f1']
        p_plus = 0 if round(sum(f1_list) / len(f1_list), 4) > 0 else 3
        if weights is not None:
            weights_mult = []
            for i, (p, r) in enumerate(weights):
                slice = torch.zeros(
                    size=(1, targets.shape[-2], targets.shape[-1]),
                    dtype=inputs.dtype, device=inputs.device
                )
                slice[targets[i] > 0] = p + p_plus
                slice[targets[i] == 0] = r
                weights_mult.append(slice)
            weights_mult = torch.stack(weights_mult, dim=0)
        else:
            weights_mult = targets.clone() + 1
        return weights_mult

    def forward(self, inputs, targets, weights=None, epoch=0):
        targets = targets.clone()
        targets[targets >= .5] = 1
        targets[targets < .5] = 0
        targets = self.decrease_y_max(targets)

        self.append_calculated_statistics(inputs=inputs, targets=targets)
        weights_mult = self.make_weights_mult(weights, targets, inputs)
        entropy = (weights_mult * self.entropy(inputs, targets)).mean()
        # entropy = (weights_mult*self.entropy(inputs, targets)).sum(dim=(1,2,3)).mean()
        # entropy = (weights_mult*self.entropy(inputs, targets)).mean(dim=(1, 2, 3)).sum()

        total = entropy
        self.loss_list_dict['BCE'].append(entropy.item())
        self.loss_list_dict['total'].append(total.item())

        return total


#
class WaterwayLossDecW1(WaterwayLossDecreaseType):
    def __init__(self, num_factors=3, **kwargs):
        super().__init__(num_factors)
        self.name = 'WaterwayLossDec'
        self.num_factors = num_factors
        self.entropy = nn.BCELoss(reduction='none')
        self.image_totals = []
        self.loss_list_dict = {
            'BCE': [], 'total': [], 'a_n': [], 'p_n': [], 'r_n': [],
            'f1_n': [], 'a_f': [], 'p_f': [], 'r_f': [], 'f1': []
        }

    def append_calculated_statistics(self, inputs, targets, weights):
        inputs_np = inputs.detach().to('cpu').float().numpy()
        inputs_np = inputs_np[:, 0]
        inputs_np[inputs_np < .5] = 0
        inputs_np[inputs_np >= .5] = 1

        targets_np = targets.detach().to('cpu').float().numpy()
        targets_np = targets_np[:, 0]
        # targets_np[weights_np == 0] = 0
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
            print('h1')
            weights[targets == 1] *= 2.
            # weights[targets == 0] = .5
        elif r > .98 and a < .2:
            print('h2')
            weights[targets == 0] += 1
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

    def forward(self, inputs, targets, weights=None, epoch=0, **kwargs):
        weights = targets.clone()
        weights[weights == 0] = 1.
        weights[weights < 1] = 0
        targets = targets.clone()
        targets[targets != 0] = 1
        targets = self.decrease_y_max(targets)
        weights = self.decrease_y_max(weights)
        self.append_calculated_statistics(inputs=inputs, targets=targets, weights=weights)
        weights = self.make_weights_mult(weights=weights, targets=targets)
        total = (weights * self.entropy(inputs, targets)).mean()
        self.loss_list_dict['BCE'].append(total.item())
        self.loss_list_dict['total'].append(total.item())
        return total
#
# class WaterwayLossDecW1(WaterwayLossDecreaseType):
#     def __init__(self, num_factors=3, **kwargs):
#         super().__init__(num_factors)
#         self.name = 'WaterwayLossDec'
#         self.num_factors = num_factors
#         self.entropy = nn.BCELoss(reduction='none')
#         self.image_totals = []
#         self.loss_list_dict = {
#             'BCE': [], 'total': [], 'a_n': [], 'p_n': [], 'r_n': [],
#             'f1_n': [], 'a_f': [], 'p_f': [], 'r_f': [], 'f1': []
#         }
#
#     def append_calculated_statistics(self, inputs, targets, weights):
#         weights_np = weights.detach().to('cpu').float().numpy()
#         weights_np = weights_np[:, 0]
#         inputs_np = inputs.detach().to('cpu').float().numpy()
#         inputs_np = inputs_np[:, 0]
#         inputs_np[inputs_np < .5] = 0
#         inputs_np[inputs_np >= .5] = 1
#         inputs_np[weights_np == 0] = 0
#
#         targets_np = targets.detach().to('cpu').float().numpy()
#         targets_np = targets_np[:, 0]
#         targets_np[weights_np == 0] = 0
#         diff = inputs_np - targets_np
#
#         outputs = self.calculate_numpy_statistics(inputs_np, targets_np, diff)
#         a, p, r, f1, a_f, p_f, r_f, f1_f, _, _, _, _, _, _, _ = outputs
#         self.loss_list_dict['a_n'].append(a)
#         self.loss_list_dict['p_n'].append(p)
#         self.loss_list_dict['r_n'].append(r)
#         self.loss_list_dict['f1_n'].append(f1)
#
#         self.loss_list_dict['a_f'].append(a_f)
#         self.loss_list_dict['p_f'].append(p_f)
#         self.loss_list_dict['r_f'].append(r_f)
#         self.loss_list_dict['f1'].append(f1_f if not np.isnan(f1_f) else 0)
#
#     def make_weights_mult(self, weights, targets):
#         # f1 = np.array(self.loss_list_dict['f1']).mean()
#         # r = np.array(self.loss_list_dict['r_f']).mean()
#         # a = np.array(self.loss_list_dict['a_f']).mean()
#
#         f1 = self.loss_list_dict['f1'][-1]
#         r = self.loss_list_dict['r_f'][-1]
#         a = self.loss_list_dict['a_f'][-1]
#         if (f1 == 0 or np.isnan(f1)) and a < .90:
#             print('h1')
#             weights[targets == 1] *= 2.
#             # weights[targets == 0] = .5
#         elif r > .98 and a < .2:
#             print('h2')
#             weights[targets == 0] += .5
#         return weights
#
#     def update_image_totals(self, image_entropy):
#         self.image_totals.extend(image_entropy.detach().to('cpu').float().numpy())
#         if len(self.image_totals) > 20*len(image_entropy):
#             self.image_totals = self.image_totals[len(image_entropy):]
#
#     def make_image_totals_weights(self, image_entropy):
#         image_totals_mean = np.array(self.image_totals).mean()
#         image_totals_std = np.array(self.image_totals).std()
#         top_val = image_totals_mean + 2.5*image_totals_std
#         image_weights = image_entropy.clone().detach()
#         image_weights[image_weights > top_val] = 0
#         return image_weights
#
#     def forward(self, inputs, targets, weights=None, epoch=0, **kwargs):
#         weights = targets.clone()
#         weights[weights == 0] = 1
#         weights[weights < 1] = 0
#         targets = targets.clone()
#         targets[targets != 0] = 1
#         targets = self.decrease_y_max(targets)
#         weights = self.decrease_y_max(weights)
#         weight_fat = targets.clone()
#         weight_fat[:, :, 1:] += weights[:, :, :-1]
#         weight_fat[:, :, :-1] += weights[:, :, 1:]
#         weight_fat[:, :, :, 1:] += weights[:, :, :, :-1]
#         weight_fat[:, :, :, :-1] += weights[:, :, :, 1:]
#         weights[(weights == 0) & (weight_fat > 0)] = .5
#         self.append_calculated_statistics(inputs=inputs, targets=targets, weights=weights)
#         weights = self.make_weights_mult(weights=weights, targets=targets)
#         # total = (weights * self.entropy(inputs, targets)).mean()
#         image_entropy = (weights * self.entropy(inputs, targets)).mean(dim=(1, 2, 3))
#         self.update_image_totals(image_entropy)
#         image_weights = self.make_image_totals_weights(image_entropy)
#         total = (image_weights*image_entropy).mean()
#         self.loss_list_dict['BCE'].append(total.item())
#         self.loss_list_dict['total'].append(total.item())
#
#         return total


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


class WaterwayTrailLossDecW1(WaterwayLossDecreaseType):
    def __init__(self, num_factors=3, **kwargs):
        super().__init__(num_factors)
        self.name = 'WaterwayLossDec'
        self.num_factors = num_factors
        self.entropy = nn.BCELoss(reduction='none')
        self.image_totals = []
        self.loss_list_dict = {
            'total': [],
            'a_w': [], 'p_w': [], 'r_w': [], 'f1_w': [],
            'a_t': [], 'p_t': [], 'r_t': [], 'f1_t': [],
            'f1': []
        }

    def append_calculated_statistics(self, inputs, targets, weights):
        # weights_np = weights.detach().to('cpu').float().numpy()
        # weights_np = weights_np[:, 0]
        inputs_np = inputs.detach().to('cpu').float().numpy()
        inputs_np[inputs_np < .5] = 0
        inputs_np[inputs_np >= .5] = 1
        # inputs_np[weights_np == 0] = 0

        targets_np = targets.detach().to('cpu').float().numpy()
        # targets_np[weights_np == 0] = 0
        diff = inputs_np - targets_np

        outputs = self.calculate_numpy_statistics(inputs_np[:, 0], targets_np[:, 0], diff[:, 0])
        a, p, r, f1, a_f, p_f, r_f, f1_f, _, _, _, _, _, _, _ = outputs
        self.loss_list_dict['a_w'].append(a_f)
        self.loss_list_dict['p_w'].append(p_f)
        self.loss_list_dict['r_w'].append(r_f)
        self.loss_list_dict['f1_w'].append(f1_f if not np.isnan(f1_f) else 0)
        outputs = self.calculate_numpy_statistics(inputs_np[:, 1], targets_np[:, 1], diff[:, 1])
        a, p, r, f1, a_f, p_f, r_f, f1_f, _, _, _, _, _, _, _ = outputs
        self.loss_list_dict['a_t'].append(a_f)
        self.loss_list_dict['p_t'].append(p_f)
        self.loss_list_dict['r_t'].append(r_f)
        self.loss_list_dict['f1_t'].append(f1_f if not np.isnan(f1_f) else 0)
        self.loss_list_dict['f1'].append((self.loss_list_dict['f1_t'][-1] + self.loss_list_dict['f1_w'][-1])/2)

    def make_weights_mult(self, weights, targets):
        # f1 = np.array(self.loss_list_dict['f1']).mean()
        # r = np.array(self.loss_list_dict['r_f']).mean()
        # a = np.array(self.loss_list_dict['a_f']).mean()

        f1 = self.loss_list_dict['f1'][-1]
        r = (self.loss_list_dict['r_w'][-1] + self.loss_list_dict['r_t'][-1])/2
        a = (self.loss_list_dict['a_w'][-1] + self.loss_list_dict['a_t'][-1])/2
        if (f1 == 0 or np.isnan(f1)) and a < .90:
            print('h1')
            weights[targets == 1] *= 2.
            # weights[targets == 0] = .5
        elif r > .98 and a < .2:
            print('h2')
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

    def forward(self, inputs, targets, weights=None, epoch=0, **kwargs):
        weights = targets.clone()
        weights[weights == 0] = 1
        weights[weights < 1] = 0
        weights[:, 0][(weights[:, 1] > 1) & (weights[:, 0] == 1)] = 2
        weights[:, 1][(weights[:, 1] == 1) & (weights[:, 0] > 1)] = 2

        targets = targets.clone()
        targets[targets != 0] = 1
        targets = self.decrease_y_max(targets)
        weights = self.decrease_y_max(weights)
        self.append_calculated_statistics(inputs=inputs, targets=targets, weights=weights)
        weights = self.make_weights_mult(weights=weights, targets=targets)
        bce = (weights * self.entropy(inputs, targets)).mean()
        weights1 = weights.clone()
        weights1[weights1 > 1] = 1
        tanimoto_ww = tanimoto_loss(inputs[:, 0], targets[:, 0], weights1[:, 0])
        tanimoto_trails = tanimoto_loss(inputs[:, 1], targets[:, 1], weights1[:, 1])
        # total = tanimoto_ww + tanimoto_trails
        total = bce + tanimoto_ww + tanimoto_trails
        # self.loss_list_dict['BCE'].append(bce.item())
        self.loss_list_dict['total'].append(total.item())

        return total



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
        # self.loss_list_dict = {
        #     'tanimoto':[],'total': [], 'a_n': [], 'p_n': [], 'r_n': [],
        #     'f1_n': [], 'a_f': [], 'p_f': [], 'r_f': [], 'f1': []
        # }

    def append_calculated_statistics(self, inputs, targets, weights):
        # weights_np = weights.detach().to('cpu').float().numpy()
        # weights_np = weights_np[:, 0]
        inputs_np = inputs.detach().to('cpu').float().numpy()
        inputs_np = inputs_np[:, 0]
        inputs_np[inputs_np < .5] = 0
        inputs_np[inputs_np >= .5] = 1
        # inputs_np[weights_np == 0] = 0

        targets_np = targets.detach().to('cpu').float().numpy()
        targets_np = targets_np[:, 0]
        # targets_np[weights_np == 0] = 0
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
        # f1 = np.array(self.loss_list_dict['f1']).mean()
        # r = np.array(self.loss_list_dict['r_f']).mean()
        # a = np.array(self.loss_list_dict['a_f']).mean()

        f1 = self.loss_list_dict['f1'][-1]
        r = self.loss_list_dict['r_f'][-1]
        a = self.loss_list_dict['a_f'][-1]
        if (f1 == 0 or np.isnan(f1)) and a < .90:
            print('h1')
            weights[targets == 1] *= 2.
            # weights[targets == 0] = .5
        elif r > .98 and a < .2:
            print('h2')
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

    def forward(self, inputs, targets, weights=None, epoch=0, **kwargs):
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
    def forward(self, inputs, targets, weights=None, epoch=0):
        targets = targets.clone()
        values = targets.clone()
        targets[targets > 0] = 1
        # targets[targets < .] = 0
        targets = self.decrease_y_max(targets)
        values = self.decrease_y_max(values)
        values = values.to('cpu').float().numpy()
        inputs_np = inputs.detach().to('cpu').float().numpy()
        inputs_np[inputs_np < .5] = 0
        inputs_np[inputs_np >= .5] = 1
        targets_np = targets.detach().to('cpu').float().numpy()
        # size = values.shape[-1] * values.shape[-2]
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


class WaterwayTrailLossDecForEval(WaterwayLossDecreaseType):
    def __init__(self, num_factors=3, max_target=[21, 2]):
        super().__init__(num_factors)
        self.name = 'WaterwayLossDec'
        self.entropy = nn.BCELoss(reduction='none')
        
        self.types = ['water', 'trails']
        self.lld_base_keys = [
            'a_n', 'p_n', 'r_n', 'f1_n', 'a_f', 'p_f', 'r_f', 'f1', 'num_one_correct', 'num_model_one',
            'num_target_one', 'num_correct', 'num_wrong', 'num_fp_f', 'num_fn_f'
        ]
        self.loss_list_dict = {f'{key}_{type}': [] for type in self.types for key in self.lld_base_keys}
        self.summed_values_base_keys = [
            'num_one_correct', 'num_model_one', 'num_target_one', 'num_correct', 'num_wrong', 'num_one_correct_f',
            'num_model_one_f', 'num_target_one_f', 'num_correct_f', 'num_wrong_f'
        ]
        self.summed_values = {f'{key}_{type}': 0 for type in self.types for key in self.summed_values_base_keys}
        self.pixel_stats_base_keys = [
            'a_pixel', 'p_pixel', 'r_pixel', 'f1_pixel', 'a_pixel_f', 'p_pixel_f', 'r_pixel_f', 'f1_pixel_f'
        ]
        self.pixel_stats = {f'{key}_{type}': 0. for type in self.types for key in self.pixel_stats_base_keys}
        self.max_target_list = max_target
        for type_index, type in enumerate(self.types):
            max_target = self.max_target_list[type_index] + 1
            for i in range(max_target):
                self.loss_list_dict[f'target_val_{i}_{type}'] = []

    def calculate_statistics(self, inputs_np, targets_np, type_index):
        inputs_np_all = inputs_np
        targets_np_all = targets_np
        diff_all = inputs_np_all - targets_np_all
        type = self.types[type_index]
        for i in range(inputs_np_all.shape[0]):
            inputs_np = inputs_np_all[i:i + 1]
            targets_np = targets_np_all[i:i + 1]
            diff = diff_all[i:i + 1]
            outputs = self.calculate_numpy_statistics(inputs_np, targets_np, diff)
            a, p, r, f1, a_f, p_f, r_f, f1_f, nc, nmo, nto, noc, nw, fp_f, fn_f = outputs
            self.loss_list_dict[f'a_n_{type}'].append(a)
            self.loss_list_dict[f'p_n_{type}'].append(p)
            self.loss_list_dict[f'r_n_{type}'].append(r)
            self.loss_list_dict[f'f1_n_{type}'].append(f1)

            self.loss_list_dict[f'a_f_{type}'].append(a_f)
            self.loss_list_dict[f'p_f_{type}'].append(p_f)
            self.loss_list_dict[f'r_f_{type}'].append(r_f)
            self.loss_list_dict[f'f1_{type}'].append(f1_f)

            self.loss_list_dict[f'num_one_correct_{type}'].append(noc)
            self.loss_list_dict[f'num_model_one_{type}'].append(nmo)
            self.loss_list_dict[f'num_target_one_{type}'].append(nto)
            self.loss_list_dict[f'num_correct_{type}'].append(nc)
            self.loss_list_dict[f'num_wrong_{type}'].append(nw)
            self.loss_list_dict[f'num_fp_f_{type}'].append(fp_f)
            self.loss_list_dict[f'num_fn_f_{type}'].append(fn_f)

            self.summed_values[f'num_one_correct_{type}'] += noc
            self.summed_values[f'num_model_one_{type}'] += nmo
            self.summed_values[f'num_target_one_{type}'] += nto
            self.summed_values[f'num_correct_{type}'] += nc
            self.summed_values[f'num_wrong_{type}'] += nw

            self.summed_values[f'num_one_correct_f_{type}'] += noc + fp_f
            self.summed_values[f'num_model_one_f_{type}'] += nmo
            self.summed_values[f'num_target_one_f_{type}'] += nto + fp_f - fn_f
            self.summed_values[f'num_correct_f_{type}'] += nc + fp_f + fn_f
            self.summed_values[f'num_wrong_f_{type}'] += nw - fp_f - fn_f
        self.pixel_stats[f'a_pixel_{type}'] = self.summed_values[f'num_correct_{type}']/(
                self.summed_values[f'num_correct_{type}'] + self.summed_values[f'num_wrong_{type}']
        )
        self.pixel_stats[f'p_pixel_{type}'] = self.summed_values[f'num_one_correct_{type}']/self.summed_values[f'num_model_one_{type}']
        self.pixel_stats[f'r_pixel_{type}'] = self.summed_values[f'num_one_correct_{type}']/self.summed_values[f'num_target_one_{type}']
        self.pixel_stats[f'f1_pixel_{type}'] = 2*self.pixel_stats[f'p_pixel_{type}']*self.pixel_stats[f'r_pixel_{type}']/(
                self.pixel_stats[f'p_pixel_{type}'] + self.pixel_stats[f'r_pixel_{type}']
        )

        self.pixel_stats[f'a_pixel_f_{type}'] = self.summed_values[f'num_correct_f_{type}']/(
                self.summed_values[f'num_correct_f_{type}'] + self.summed_values[f'num_wrong_f_{type}']
        )
        self.pixel_stats[f'p_pixel_f_{type}'] = self.summed_values[f'num_one_correct_f_{type}']/self.summed_values[f'num_model_one_f_{type}']
        self.pixel_stats[f'r_pixel_f_{type}'] = self.summed_values[f'num_one_correct_f_{type}']/self.summed_values[f'num_target_one_f_{type}']
        self.pixel_stats[f'f1_pixel_f_{type}'] = 2*self.pixel_stats[f'p_pixel_f_{type}']*self.pixel_stats[f'r_pixel_f_{type}']/(
                self.pixel_stats[f'p_pixel_f_{type}'] + self.pixel_stats[f'r_pixel_f_{type}']
        )

    def forward(self, inputs, targets, weights=None, epoch=0):
        targets = targets.clone()
        values = targets.clone()
        targets[targets > 0] = 1
        # targets[targets < .] = 0
        targets = self.decrease_y_max(targets)
        values = self.decrease_y_max(values)
        values = values.to('cpu').float().numpy()
        size = values.shape[-1]*values.shape[-2]
        for type_index, type in enumerate(self.types):
            max_target = self.max_target_list[type_index] + 1
            for i in range(max_target):
                for j in range(values.shape[0]):
                    slice = values[j, type_index]
                    self.loss_list_dict[f'target_val_{i}_{type}'].append(100*len(np.where(slice == i)[0])/size)

        inputs_np = inputs.detach().to('cpu').float().numpy()
        inputs_np[inputs_np < .5] = 0
        inputs_np[inputs_np >= .5] = 1

        targets_np = targets.detach().to('cpu').float().numpy()
        self.calculate_statistics(inputs_np=inputs_np[:, 0], targets_np=targets_np[:, 0], type_index=0)
        self.calculate_statistics(inputs_np=inputs_np[:, 1], targets_np=targets_np[:, 1], type_index=1)

