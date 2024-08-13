import PyQt6
# import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import TextBox, RadioButtons
import numpy as np
from water.paths import ppaths
import torch
from torch import nn
from torch.nn import functional as F

def decrease_y(y, num_factors):
    for i in range(num_factors):
        y = F.max_pool2d(y, 2)
    return y

# def decrease_y(y, num_factors):
#     y = F.max_pool2d(y, 2)
#     y = F.interpolate(y, scale_factor=2, mode='bicubic', align_corners=True)
#     y = F.max_pool2d(y, 2)
#     y = F.max_pool2d(y, 2)
#     return y



class MyPlot:
    def __init__(self, batches=(1,2,3),
                 model_index=0,
                 model_increase=0.,
                 num_decrease_steps: int = None):
        self.raw = None
        self.model = None
        for batch in batches:
            if self.raw is None:
                self.raw = np.load(ppaths.training_data/f'model_results/training_batch_{batch}/raw.npy').astype(np.float32)
                self.model = np.load(ppaths.training_data/f'model_results/training_batch_{batch}/y_model.npy')[:,
                             model_index].astype(np.float32)
            else:
                raw = np.load(ppaths.training_data/f'model_results/training_batch_{batch}/raw.npy').astype(np.float32)
                model = np.load(ppaths.training_data/f'model_results/training_batch_{batch}/y_model.npy')[:,
                             model_index].astype(np.float32)
                self.raw = np.concatenate([self.raw, raw], axis=0)
                self.model = np.concatenate([self.model, model], axis=0)
        print(self.raw.shape)
        self.burned = self.raw[:, -1:].copy()
        # self.burned[self.burned > 0] = 1
        print(self.model.shape, self.burned.shape)
        if num_decrease_steps is None:
            num_decrease_steps = int(self.burned.shape[-1]/self.model.shape[-1]) - 1
            print(num_decrease_steps)
        if num_decrease_steps > 0:
            self.burned = torch.tensor(self.burned)
            print(self.burned.shape)
            self.burned = decrease_y(self.burned, num_decrease_steps)
            self.burned = self.burned.detach().numpy()
        # self.model = torch.tensor(self.model)
        # self.burned = torch.tensor(self.burned)
        # self.burned = decrease_y(self.burned, 2).detach().numpy()
        # self.model = decrease_y(self.model, 2).detach().numpy()
        self.burned = self.burned[:, 0]
        print(self.model.shape, self.burned.shape)
        self.burned_1 = self.burned.copy()
        self.burned_1[self.burned_1 >= .5] = 1
        self.burned_1[self.burned_1 < .5] = 0

        # self.diff = self.model - self.burned_1
        self.model = self.model + model_increase
        self.ele = self.raw[:, -2].copy()
        self.elevation = self.raw[:, -5].copy()
        self.rounded_diff = np.round(self.model) - np.round(self.burned_1)
        self.diff = self.rounded_diff.copy()
        self.ele = np.round(self.model)
        self.raw_s = (self.raw[:, 1:4].copy() + 1)/2
        self.raw_s = np.transpose(self.raw_s, (0, 2, 3, 1))
        self.titles = {}
        self.current_diff_label = 'real'
        # self.sen_diff = np.abs(self.x_up - self.sen_up).mean(axis=3)*255
        self.num_grids = len(self.raw_s)
        self.fig, self.ax = plt.subplots(2, 3, sharex=True, sharey=True)
        # self.fig, self.ax = plt.subplots(2, 2, sharex=True, sharey=True)
        
        self.plots_dict = {}
        self.grid_num = 0
        self.plot()
    
    def update_plots(self):
        for key in self.plots_dict:
            data = self.plots_dict[key]['data']
            plot = self.plots_dict[key]['image']
            plot.set_data(data[self.grid_num])
        self.fig.canvas.draw()
        self.get_precision_recall_accuracy()
    
    def update_grid_num(self, new_num):
        try:
            new_num = int(float(new_num))
            new_num = max(new_num, 0)
            new_num = min(self.num_grids, new_num)
            print(self.elevation[new_num].sum())
            self.grid_select.set_val(new_num)
            self.grid_num = new_num
            self.update_plots()
            # self.ax[3].set_title(f'L1 Diff (avg: {self.diff_by_image[new_num]:.4f})')
        except ValueError:
            new_num = int(self.grid_num)
            self.grid_select.set_val(new_num)
        #     print('here')
    
    def add_title(self, ax, label, name=None):
        if name is None:
            name = label
        txt = ax.text(
                .5, 1, label,
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes
        )
        self.titles[name] = txt
    
    def update_title(self, name, new_text):
        title = self.titles[name]
        title.set_text(new_text)
    
    def get_precision_recall_accuracy(self, plot_total=False):
        if plot_total:
            cur_diff = self.rounded_diff
            cur_model = np.round(self.model)
            cur_burned = np.round(self.burned_1)
            num_one_correct = len(cur_model[(cur_model == 1) & (cur_burned == 1)])
            num_model_one = len(cur_model[(cur_model == 1)])
            if num_model_one == 0:
                num_model_one = 1
            num_burned_one = len(cur_burned[(cur_burned == 1)])
            if num_burned_one == 0:
                num_burned_one = 1
            fp_w1, total_fp = self.get_within_n_all(cur_model, cur_burned, 1, 1)
            fn_w1, total_fn = self.get_within_n_all(cur_model, cur_burned, 0, 1)
            if total_fp == 0:
                total_fp = 1
            if total_fn == 0:
                total_fn = 1
            print(f'Stats for all {len(cur_diff)} images:')
            accuracy = len(cur_diff[cur_diff == 0])/(cur_diff.shape[0]*cur_diff.shape[-1]*cur_diff.shape[-2])
            print(f' accuracy: {accuracy:.2%}')
            print(f'precision: {num_one_correct/num_model_one:.2%}')
            print(f'   recall: {num_one_correct/num_burned_one:.2%}')
            print(f'precision1: {(num_one_correct + fp_w1)/num_model_one:.2%}')
            print(f'   recall1: {(num_one_correct + fp_w1)/(max(num_burned_one - fn_w1 + fp_w1, 1)):.2%}')
            print(f'per fp w1: {fp_w1/total_fp:.2%} ({fp_w1}/{total_fp})')
            print(f'per fn w1: {fn_w1/total_fn:.2%} ({fn_w1}/{total_fn})')
            print('')
        
        cur_diff = self.rounded_diff[self.grid_num]
        cur_model = np.round(self.model[self.grid_num])
        cur_burned = np.round(self.burned_1[self.grid_num])
        num_one_correct = len(cur_model[(cur_model == 1) & (cur_burned == 1)])
        num_model_one = len(cur_model[(cur_model == 1)])
        if num_model_one == 0:
            num_model_one = 1
        num_burned_one = len(cur_burned[(cur_burned == 1)])
        if num_burned_one == 0:
            num_burned_one = 1
        fp_w1, total_fp = self.get_within_n_single(cur_model, cur_burned, 1, 1)
        fn_w1, total_fn = self.get_within_n_single(cur_model, cur_burned, 0, 1)
        if total_fp == 0:
            total_fp = 1
        if total_fn == 0:
            total_fn = 1
        print(f'Stats for image {self.grid_num}:')
        print(f' accuracy: {len(cur_diff[cur_diff == 0])/(cur_diff.shape[-1]*cur_diff.shape[-2]):.2%}')
        print(f'precision: {num_one_correct/num_model_one:.2%}')
        print(f'   recall: {num_one_correct/num_burned_one:.2%}')
        print(f'precision1: {(num_one_correct + fp_w1)/num_model_one:.2%}')
        print(f'   recall1: {(num_one_correct + fp_w1)/(max(num_burned_one - fn_w1 + fp_w1, 1)):.2%}')
        print(f'per fp w1: {fp_w1/total_fp:.2%} ({fp_w1}/{total_fp})')
        print(f'per fn w1: {fn_w1/total_fn:.2%} ({fn_w1}/{total_fn})')
        print('')
    
    def get_within_n_single(self, model, burned, model_val, n):
        rows, cols = np.where((model == model_val) & (model != burned))
        other_val = abs(model_val - 1)
        max_rows = model.shape[-2]
        max_cols = model.shape[-1]
        total = len(rows)
        within_n = 0
        for row, col in zip(rows, cols):
            rm = max(row - n, 0)
            rM = min(row + n + 1, max_rows)
            cm = max(col - n, 0)
            cM = min(col + n + 1, max_cols)
            if model_val in burned[rm:rM, cm:cM] and other_val in model[rm:rM, cm:cM]:
                within_n += 1
        return within_n, total
    
    def get_within_n_all(self, model, burned, model_val, n):
        ex, rows, cols = np.where((model == model_val) & (model != burned))
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
            if model_val in burned[ex, rm:rM, cm:cM] and other_val in model[ex, rm:rM, cm:cM]:
                within_n += 1
                self.diff[ex, row, col] = 0
        return within_n, total
    
    def plot(self):
        self.get_precision_recall_accuracy(True)
        
        r, c, _ = self.raw_s[0].shape
        sen = self.ax[0, 0].imshow(self.raw_s[0], extent=(0, c, 0, r))
        # self.ax[0, 0].set_title('Sentinel Data', fontdict={'fontsize':10})
        # self.add_title(self.ax[0, 0], 'Sentinel Data')
        
        # elevation = self.ax[0, 1].imshow(self.ele[0], cmap='inferno', vmin=0, vmax=self.ele.mean()+self.ele.std(), extent=(0,c,0,r))
        elevation = self.ax[1, 1].imshow(self.ele[0], cmap='binary', vmin=0, vmax=1, extent=(0, c, 0, r))
        
        # self.add_title(self.ax[1, 0], 'elevation')
        
        burned = self.ax[1, 0].imshow(self.burned[0], vmin=0, vmax=3, cmap='binary', extent=(0, c, 0, r))
        # self.add_title(self.ax[0, 1], 'burned')
        
        model = self.ax[0, 1].imshow(self.model[0], vmin=0, vmax=1, cmap='binary', extent=(0, c, 0, r))
        # self.add_title(self.ax[1, 1], 'model_outputs')
        
        diff = self.ax[1, 2].imshow(self.diff[0], vmin=-1, vmax=1, cmap='PiYG', extent=(0, c, 0, r))
        # self.add_title(self.ax[0, 2], 'diff')
        
        rounded_diff = self.ax[0, 2].imshow(self.rounded_diff[0], vmin=-1, vmax=1, cmap='PiYG', extent=(0, c, 0, r))
        # self.add_title(self.ax[1, 2], 'rounded_diff')
        r, c, _ = self.raw_s[0].shape
        
        # burned = self.ax[0, 0].imshow(self.burned[0], vmin=0, vmax=1, cmap='binary', extent=(0,c,0,r))
        # self.add_title(self.ax[0, 0], 'target data')
        #
        # model = self.ax[0, 1].imshow(self.model[0], vmin=.3, vmax=.5, cmap='binary', extent=(0,c,0,r))
        # self.add_title(self.ax[0, 1], 'model_outputs')
        #
        # diff = self.ax[1, 1].imshow(self.diff[0], vmin=-1, vmax=1, cmap='PiYG', extent=(0,c,0,r))
        # self.add_title(self.ax[1, 1], 'difference (nearby removed)')
        #
        # rounded_diff = self.ax[1, 0].imshow(self.rounded_diff[0], vmin=-1, vmax=1, cmap='PiYG', extent=(0,c,0,r))
        # self.add_title(self.ax[1, 0], 'proper difference')
        #
        # self.plots_dict = {'burned': {'data': self.burned, 'image': burned},
        #                    'model': {'data': self.model, 'image': model},
        #                    'diff': {'data': self.diff, 'image': diff},
        #                    'rounded_diff':{'data':self.rounded_diff, 'image':rounded_diff},
        #                    }
        self.plots_dict = {'sen': {'data': self.raw_s, 'image': sen},
                           'elevation': {'data': self.ele, 'image': elevation},
                           'burned': {'data': self.burned, 'image': burned},
                           'model': {'data': self.model, 'image': model},
                           'diff': {'data': self.diff, 'image': diff},
                           'rounded_diff': {'data': self.rounded_diff, 'image': rounded_diff},
                           }
        for m in range(3):
            for n in range(2):
                self.ax[n, m].xaxis.set_ticklabels([])
                self.ax[n, m].xaxis.set_ticks([])
                self.ax[n, m].yaxis.set_ticklabels([])
                self.ax[n, m].yaxis.set_ticks([])
        
        plt.subplots_adjust(
                top=0.98,
                bottom=0.04,
                left=0.13,
                right=0.87,
                hspace=0.02,
                wspace=0.09
        )
        text_box_height = 0.03
        height = .6
        axbox = self.fig.add_axes([0.0625, .01, 0.08, text_box_height])
        self.grid_select = TextBox(ax=axbox,
                                   label_pad=.1,
                                   label=f'Grid (0-{self.num_grids-1}):',
                                   initial=f'{int(self.grid_num)}',
                                   textalignment='center'
                                   )
        self.grid_select.on_submit(self.update_grid_num)
        # diff_ax = self.fig.add_axes([.955, .4, .035, .2])
        # self.diff_select = RadioButtons(ax=diff_ax,
        #                                 labels=['og', 'vs_s', 'vs_n'])
        # self.diff_select.on_clicked(self.update_diff)


class TrainingLogPlot:
    def __init__(self, epochs=None, model_index=None, test_index=1000000,
                 plot_test=False, base_path=ppaths.training_data/'model_data', index_div=1
                 ):
        self.base_path = base_path
        if model_index is None:
            model_index = self.get_max_model_index()
        print(model_index)
        self.model_index = model_index
        self.model_dir = base_path/f'model_{model_index}/training_logs'
        epoch_files = self.get_epoch_files(epochs)
        if plot_test:
            self.training_files = [file for file in epoch_files if self.get_log_data_index(file) >= test_index]
        else:
            self.training_files = [file for file in epoch_files if self.get_log_data_index(file) < test_index]
        self.training_files.sort(key=self.get_log_data_index)
        self.training_df = pd.concat([pd.read_csv(file) for file in self.training_files])
        if not plot_test:
            self.training_df['data_index'] = self.training_df['data_index']/index_div - 1
        print(self.training_df)
    
    def plot_histogram(self, column, min_n=None, max_n=0, ax=None):
        to_plot = self.training_df
        print(len(to_plot))
        max_val = to_plot.data_index.max() + 1
        if min_n is None:
            max_n = max_val
        # print(max_val-max_n, max_val-max_n)
        to_plot = self.training_df[(max_val-min_n <= to_plot.data_index)
                                   & (to_plot.data_index <= max_val-max_n)].reset_index(drop=True)
        print(len(to_plot))
        if min_n is not None:
            to_plot = to_plot[to_plot.data_index > to_plot.data_index.max()-min_n]
        print(len(to_plot))
        if ax is None:
            fig, ax = plt.subplots()
        sns.histplot(data=to_plot, x=column, ax=ax, stat='probability')
        return ax
    
    def plot_col(self, column, ax=None, color='blue'):
        to_plot = self.training_df.groupby(['epoch', 'data_index', 'save_index']).mean().reset_index().reset_index()
        print(to_plot[['index', column]].corr())
        ax = to_plot.plot.scatter(x='index', y=column, ax=ax, c=color)
        return ax
    
    def get_log_data_index(self, file):
        index = int(file.name.split('_')[-2])
        return index
    
    def get_max_model_index(self):
        files = self.base_path.glob('model_*')
        max_ind = 0
        for file in files:
            ind = int(file.name.split('_')[-1])
            if ind > max_ind:
                max_ind = ind
        return max_ind
    
    def get_max_epoch(self):
        files = self.model_dir.glob('*')
        max_epoch = 0
        for file in files:
            epoch = int(file.name.split('_')[1])
            if epoch > max_epoch:
                max_epoch = epoch
        print(max_epoch)
        return max_epoch
    
    def get_epoch_files(self, epochs):
        if epochs is None:
            epochs = list(range(self.get_max_epoch() + 1))
        print(epochs)
        epoch_files = []
        if type(epochs) == list:
            for epoch in epochs:
                epoch_files += list(self.model_dir.glob(f'epoch_{epoch}_*'))
        elif type(epochs) == int:
            epoch_files = list(self.model_dir.glob(f'epoch_{epochs}_*'))
        return epoch_files

import shapely
import geopandas as gpd
def name_to_shapely_box(file_name):
    box_info = [float(item) for item in file_name.split('/')[0].split('bbox_')[-1].split('_')]
    return shapely.box(*box_info)

if __name__ == '__main__':
    import seaborn as sns
    from water.basic_functions import printdf


    # inds_to_plot = [i for i in range(1, 9) if i not in [3,4,5,6]] + [9]
    # inds_to_plot = [1,2,3,4]
    # #
    start = 9
    num = 1
    # inds_to_plot = [(start-i-1) % 9 + 1 for i in range(0, num)]
    inds_to_plot = [(start-i-1) % 8 + 1 for i in range(0, num)]
    inds_to_plot = [9]
    my_plt = MyPlot(
        batches=inds_to_plot, model_increase=.0, num_decrease_steps=1
    )
    # my_plt = MyPlot(
    #     batches=inds_to_plot, model_increase=.0, num_decrease_steps=2
    # )


    # m_index = 452
    # m_index = None
    # # #
    # tlp = TrainingLogPlot(model_index=m_index, plot_test=True, test_index=10000000, index_div=2000)
    # ax=None
    # ax = tlp.plot_histogram('wwm_f1', min_n=40, max_n=20, ax=ax)
    # ax = tlp.plot_histogram('wwm_f1', min_n=20, max_n=0, ax=ax)

    # # # print(tlp.training_df)
    # # #
    #
    # ax = tlp.plot_col('wwm_p_f', color='tab:red')
    # tlp.plot_col('wwm_r_f', color='tab:blue', ax=ax)
    # tlp.plot_col('wwm_a_f', color='tab:green', ax=ax)
    # tlp.plot_col('wwm_f1', color='tab:orange', ax=ax)
    # tlp.plot_col('wwm_BCE', color='green')
    # printdf(tlp.training_df[['wwm_BCE', 'wwm_a_f', 'wwm_p_f', 'wwm_r_f', 'wwm_f1', 'data_index']].corr(), 100)
    # from pathlib import Path
    # base_path = Path('/ilus/data/waterway_data/model_data/model_258/training_logs')
    # for path in base_path.glob('epoch_1_data_index_1000*_waterways.csv'):
    #     old_name = path.name
    #     index = int(old_name.split('epoch_1_data_index_')[-1].split('_waterways.csv')[0])
    #     if index >= 1000000:
    #         new_index = index + 68
    #         new_name = old_name.replace(f'{index}', f'{new_index}')
    #         # print(old_name, new_name)
    #         new_path = path.parent/new_name
    #         path.rename(new_path)


    # m_index = 520
    # denom = 12
    # # # # #
    # tlp = TrainingLogPlot(model_index=m_index, plot_test=False, test_index=1000000, index_div=2000)
    # tlt = TrainingLogPlot(model_index=m_index, plot_test=True, test_index=1000000, index_div=2000)
    # df = tlp.training_df
    # dfm = df.groupby('data_index').mean().reset_index().reset_index()
    # dft = tlt.training_df
    # dftm = dft.groupby('data_index').mean().reset_index().reset_index()
    # #
    # dfm['half_epoch'] = np.floor(dfm['index']/denom)
    # dftm['half_epoch'] = np.floor(dftm['index']/denom)
    # #
    # # print(len(dfm), len(dftm))
    # cols = ['wwm_p_f', 'wwm_r_f', 'wwm_f1']
    # # cols = ['wwm_BCE']
    #
    # # #
    # fig, ax = plt.subplots()
    #
    # df_list = []
    # for df, color in [(dfm, 'train'), (dftm, 'test')]:
    #     for col in cols:
    #         dfi = df[['half_epoch']].copy()
    #         dfi['val'] = df[col]
    #         dfi['metric'] = col
    #         dfi['train_test'] = color
    #         df_list.append(dfi)
    # df1 = pd.concat(df_list, ignore_index=True)
    # sns.lineplot(ax=ax, data=df1, x='half_epoch', y='val', style='train_test', hue='metric')

    # ax = tlp.training_df.groupby(['data_index', 'save_index']).mean().reset_index().reset_index().plot.scatter(x='index', y='waterway_model_accuracy', c='red')
    # tlp.training_df.groupby(['data_index', 'save_index']).mean().reset_index().reset_index().plot.scatter(x='index', y='waterway_model_precision', ax=ax, c='green')
    # tlp.training_df.groupby(['data_index', 'save_index']).mean().reset_index().reset_index().plot.scatter(x='index', y='waterway_model_recall', ax=ax, legend=True, c='blue')
