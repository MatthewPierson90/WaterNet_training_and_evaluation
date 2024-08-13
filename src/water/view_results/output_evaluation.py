import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import rasterio as rio
import matplotlib
from matplotlib.widgets import Button
from water.basic_functions import open_json, save_json
from water.paths import ppaths
import rioxarray as rxr
from water.data_functions.load.load_waterway_data import SenElBurnedLoader, SenBurnedLoader
from water.training.test_model import get_input_files
import geopandas as gpd
from pathlib import Path
import shapely
from torch.nn import functional as F
from collections import defaultdict

def decrease_y(y, num_factors):
    y = torch.tensor(np.stack([y]))
    for i in range(num_factors):
        y = F.max_pool2d(y, 2)
    y = y.numpy()[0]
    return y

def open_hu4_parquet_data(index):
    return gpd.read_parquet(ppaths.training_data/f'hu4_parquet/hu4_{index:04d}.parquet')

def make_input_output_files(output_dir: Path, input_files: list[Path]):
    output_files = [output_dir/file.parent.name/file.name for file in input_files]
    return list(zip(input_files, output_files))


class Chooser:
    def __init__(self, model_number, clear_seen=False,
                 eval_files=None, path_pairs=None,
                 eval_dir=ppaths.training_data/'model_inputs_224',
                 output_dir=None,
                 data_loader: SenBurnedLoader = SenBurnedLoader(),
                 decrease_steps=2,
                 values_to_ignore=(1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 15),
                 ):
        count = 0
        self.decrease_steps = decrease_steps
        self.data_loader = data_loader
        self.eval_dir = eval_dir
        self.seen_path = eval_dir/f'good_bad_ugly_model_eval.json'
        self.hu4_hulls = gpd.read_parquet(ppaths.training_data/'hu4_hulls.parquet')
        self.hull_tree = shapely.STRtree(self.hu4_hulls.geometry.to_list())
        self.index_to_hu4_index = {ind: idx for ind, idx in enumerate(self.hu4_hulls.hu4_index)}
        self.current_hu4 = 0
        self.values_to_ignore = values_to_ignore
        self.hu4_gdf = open_hu4_parquet_data(101)
        if not self.seen_path.exists() or clear_seen:
            self.seen_names = {}
        else:
            self.seen_names = open_json(self.seen_path)
        if eval_files is None:
            eval_files = get_input_files(eval_dir=eval_dir)
        self.rsl = None
        self.sen_im = None
        if output_dir is None:
            output_dir = eval_dir/f'output_data_{model_number}'
        self.path_pairs = path_pairs
        if path_pairs is None:
            self.path_pairs = make_input_output_files(output_dir=output_dir, input_files=eval_files)
        print(len(self.path_pairs))
        # np.random.shuffle(self.path_pairs)
        self.current_index = 0
        self.fig, self.ax = plt.subplots(2, 3, sharex=True, sharey=True)
        self.diff_cmap = matplotlib.colors.ListedColormap(['#fa6c00', '#ffffff', '#001dfa'])
        # self.fig, self.ax = plt.subplots(2, 3, sharex=False, sharey=False)

        # self.ax[0, 0].ticklabel_format(useOffset=False, style='plain')
        # self.ax[0, 0].get_xaxis().get_major_formatter().set_scientific(False)
        self.ax[0, 0].set_title('a.)', loc='left')
        self.ax[0, 1].set_title('b.)', loc='left')
        self.ax[0, 2].set_title('c.)', loc='left')
        self.ax[1, 0].set_title('e.)', loc='left')
        self.ax[1, 1].set_title('f.)', loc='left')
        self.ax[1, 2].set_title('g.)', loc='left')

        # self.ax[0, 1].set_title('NHD Data')
        # self.ax[0, 2].set_title('Rounded Model Output Minus NHD Data')
        # self.ax[1, 0].set_title('Model Output')
        # self.ax[1, 1].set_title('Model Output Rounded at 0.5')
        # self.ax[1, 2].set_title('Ignoring Nearby Errors')
        # self.ax[0, 0].set_title('Sentinel 2 Input')
        # self.ax[0, 1].set_title('NHD Data')
        # self.ax[0, 2].set_title('Rounded Model Output Minus NHD Data')
        # self.ax[1, 0].set_title('Model Output')
        # self.ax[1, 1].set_title('Model Output Rounded at 0.5')
        # self.ax[1, 2].set_title('Ignoring Nearby Errors')

        for i in range(2):
            for j in range(3):
                self.ax[i, j].ticklabel_format(useOffset=False, style='plain')
                self.ax[i, j].get_xaxis().get_major_formatter().set_scientific(False)
        # for i in range(2):
        #     for j in range(3):
        #         if i != 0 or j != 0:
        #             self.remove_ticks_from_axis(self.ax[i, j])
        # self.ax[1, 0].axis('off')

        # for i in [0, 1]:
        #     self.remove_ticks_from_axis(self.ax[i])
        plt.subplots_adjust(
            top=0.975,
            bottom=0.044,
            left=0.09,
            right=0.885,
            hspace=0.03,
            wspace=0.025
        )
        self.make_plot()
    
    def open_data(self):
        input_path, output_path = self.path_pairs[self.current_index]
        # print(f'{input_path.parent.name}/{input_path.name}')
        # print(f'{output_path.parent.name}/{output_path.name}')
        with rio.open(output_path) as rio_f:
            # output = np.round(rio_f.read()[0])
            output = rio_f.read()[0]

            bbox = tuple(rio_f.bounds)
        
        sen_ds = self.data_loader.open_data(input_path)
        sen_data = sen_ds[1:4]
        sen_data = (sen_data.transpose((1, 2, 0)) + 1)/2
        gradient = sen_ds[-2]
        burned = sen_ds[-1]
        burned = decrease_y(burned, self.decrease_steps)
        print(burned.max())
        burned_ignored = burned.copy()
        output_ignored = output.copy()
        to_ignore = np.isin(burned, self.values_to_ignore)
        output_ignored[to_ignore] = 0
        burned_ignored[to_ignore] = 0
        burned_ignored[burned_ignored > 1] = 1
        burned[(burned > 1)] = 1
        # burned[21 == burned] = 2
        self.burned_ignored = burned_ignored
        self.output_ignored = np.round(output_ignored)
        # waterways = gpd.read_parquet(ww_path)
        waterways = None
        return sen_data, gradient, burned, output, waterways, bbox
    
    def update_plots(self, init=False):
        if not init:
            self.current_index += 1
        text = f'{self.current_index:{len(str(len(self.path_pairs)))}d}/{len(self.path_pairs)}'
        self.ci_text.set_text(text)
        sen_data, gradient, burned, output, ww, bbox = self.open_data()
        self.model = output
        self.model_rounded = np.round(output)
        self.burned = burned
        self.diff = self.model_rounded - burned
        self.diff_ignored = self.output_ignored - self.burned_ignored
        self.diff_fixed = self.diff.copy()
        self.bbox = bbox
        self.print_image_stats()
        if self.sen_im is None:
            self.sen_im = self.ax[0, 0].imshow(sen_data, extent=(bbox[0], bbox[2], bbox[1], bbox[3]))
            # self.plot_ww(ax=self.ax[0], intersects_df=intersects_box)
            # self.ax[0, 0].set_aspect(1)
            self.naip_im = self.ax[0, 1].imshow(
                    burned, vmax=1, vmin=0, extent=(bbox[0], bbox[2], bbox[1], bbox[3]), cmap='binary'
            )
            self.out_im_rounded = self.ax[1, 1].imshow(
                    self.model_rounded, vmax=.75, vmin=.25, extent=(bbox[0], bbox[2], bbox[1], bbox[3]), cmap='binary'
            )
            self.out_im = self.ax[1, 0].imshow(
                    output, vmax=.95, vmin=.05, extent=(bbox[0], bbox[2], bbox[1], bbox[3]), cmap='binary'
            )
            self.diff_im = self.ax[0, 2].imshow(
                self.diff, vmin=-1, vmax=1, extent=(bbox[0], bbox[2], bbox[1], bbox[3]), cmap=self.diff_cmap
            )
            # self.diff_fixed_im = self.ax[1, 2].imshow(
            #     self.diff_ignored, vmin=-1, vmax=1, extent=(bbox[0], bbox[2], bbox[1], bbox[3]), cmap=self.diff_cmap
            # )
            self.diff_fixed_im = self.ax[1, 2].imshow(
                self.diff_fixed, vmin=-1, vmax=1, extent=(bbox[0], bbox[2], bbox[1], bbox[3]), cmap=self.diff_cmap
            )

        else:
            self.ax[0, 0].clear()
            self.ax[0, 0].set_title('a.)', loc='left')
            self.sen_im = self.ax[0, 0].imshow(sen_data, extent=(bbox[0], bbox[2], bbox[1], bbox[3]))
            self.sen_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
            # self.ax[0, 0].set_aspect(1)
            # self.ax[0] = ww.plot(ax=self.ax[0], alpha=.5)
            # self.plot_ww(ax=self.ax[0], intersects_df=intersects_box)
            # self.ax[0, 0].set_aspect(1)
            self.naip_im.set_data(burned)
            self.naip_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
            self.out_im_rounded.set_data(self.model_rounded)
            self.out_im_rounded.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
            self.out_im.set_data(output)
            self.out_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
            self.diff_im.set_data(self.diff)
            self.diff_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
            self.diff_fixed_im.set_data(self.diff_fixed)
            # self.diff_fixed_im.set_data(self.diff_ignored)
            self.diff_fixed_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
        for i in range(2):
            for j in range(3):
                if i != 1 or j != 0:
                    self.ax[i, j].ticklabel_format(useOffset=False, style='plain')
                    self.ax[i, j].get_xaxis().get_major_formatter().set_scientific(False)
        # self.ax[0, 0].ticklabel_format(useOffset=False, style='plain')
        # self.ax[0, 0].get_xaxis().get_major_formatter().set_scientific(False)
        # self.ax[1, 0].axis('off')
        plt.draw()
        plt.show()
        self.print_image_stats()

    def plot_ww_single(self, ax, df, color):
        if len(df)>0:
            df.plot(ax=ax, color=color, alpha=0.5)
    
    def plot_ww(self, ax, intersects_df: gpd.GeoDataFrame):
        swamps_df = intersects_df[intersects_df.fcode_description.str.contains('Swamp')]
        self.plot_ww_single(ax=ax, df=swamps_df, color='red')
        playa_df = intersects_df[intersects_df.fcode_description.str.contains('Playa')]
        self.plot_ww_single(ax=ax, df=playa_df, color='orange')
        canal_df = intersects_df[intersects_df.fcode_description.str.contains('Canal')]
        self.plot_ww_single(ax=ax, df=canal_df, color='green')
        wash_df = intersects_df[intersects_df.fcode_description.str.contains('Wash')]
        self.plot_ww_single(ax=ax, df=wash_df, color='blue')
        wawa_df = intersects_df[((intersects_df.fcode_description.str.contains('Stream'))
                                 | (intersects_df.fcode_description.str.contains('Artificial'))
                                 | (intersects_df.fcode_description.str.contains('Lake'))
                                 )]
        self.plot_ww_single(ax=ax, df=wawa_df, color='aqua')
        wash_df = intersects_df[intersects_df.fcode_description.str.contains('Reservoir')]
        self.plot_ww_single(ax=ax, df=wash_df, color='yellow')
    
    def next_press(self, press, keep=True):
        file_path, _ = self.path_pairs[self.current_index]
        key = file_path.name.split('.tif')[0]
        if keep:
            self.seen_names[key] = {
                'precision': 2, 'recall': 1
            }
            save_json(self.seen_path, self.seen_names)
        # else:
        #     self.seen_names[key] = {'precision': 0,
        #                             'recall': 0}
        self.update_plots()
    
    def get_within_n_single(self, model, burned, model_val, n):
        rows, cols = np.where((model == model_val) & (model != burned))
        other_val = abs(model_val-1)
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
                self.diff_fixed[row, col] = 0
        return within_n, total


    def print_image_stats(self):
        cur_model = np.round(self.model)
        cur_burned = np.round(self.burned)
        cur_diff = cur_model - cur_burned
        num_one_correct = len(cur_model[(cur_model == 1) & (cur_burned == 1)])
        num_model_one = len(cur_model[(cur_model == 1)])
        if num_model_one == 0:
            num_model_one = 1
        num_burned_one = len(cur_burned[(cur_burned == 1)])
        if num_burned_one == 0:
            num_burned_one = 1
        accuracy = len(cur_diff[cur_diff==0])/(cur_diff.shape[-1]*cur_diff.shape[-2])
        
        # self.accuracy.set_text(f' accuracy: {accuracy:.2%}')
        p = num_one_correct/num_model_one
        r = num_one_correct/num_burned_one
        f1 = 2*p*r/(p + r) if p + r > 0 else 0
        self.standard_scores.set_text(self.make_text(p, r, f1, 0))

        fp_w1, total_fp = self.get_within_n_single(cur_model, cur_burned, 1, 1)
        fn_w1, total_fn = self.get_within_n_single(cur_model, cur_burned, 0, 1)
        p = (num_one_correct)/(num_model_one-fp_w1)
        r = (num_one_correct)/(max(num_burned_one - fn_w1, 1))
        f1 = 2*p*r/(p + r) if p + r > 0 else 0
        # cur_model = self.output_ignored
        # cur_burned = self.burned_ignored
        # num_one_correct = len(cur_model[(cur_model == 1) & (cur_burned == 1)])
        # num_model_one = len(cur_model[(cur_model == 1)])
        # if num_model_one == 0:
        #     num_model_one = 1
        # num_burned_one = len(cur_burned[(cur_burned == 1)])
        # if num_burned_one == 0:
        #     num_burned_one = 1
        # p = num_one_correct/num_model_one
        # r = num_one_correct/num_burned_one
        # f1 = 2*p*r/(p + r) if p + r > 0 else 0
        self.ignored_scores.set_text(self.make_text(p, r, f1, 1))
        return p, r, f1, accuracy
    
    def back_press(self, press):
        self.current_index -= 2
        # file_path = self.path_pairs[self.current_index]
        
        self.update_plots()
    
    def reset_press(self, press):
        left, bottom, right, top = self.bbox
        width = right - left
        height = top - bottom
        # self.ax[0].set_position((left, bottom, width, height))
        for i in range(2):
            for j in range(2):

                self.ax[i, j].xaxis._set_lim(v0=left, v1=right, auto=True)
                self.ax[i, j].yaxis._set_lim(v0=bottom, v1=top, auto=True)
                self.ax[i, j].set_aspect(1)
        # self.ax[1].xaxis._set_lim(v0=left, v1=right, auto=True)
        # self.ax[1].yaxis._set_lim(v0=bottom, v1=top, auto=True)
        # self.ax[1].set_aspect(1)
        # self.ax[2].xaxis._set_lim(v0=left, v1=right, auto=True)
        # self.ax[2].yaxis._set_lim(v0=bottom, v1=top, auto=True)
        # self.ax[2].set_aspect(1)
        # self.ax[1].set_position((left, bottom, width, height))
        plt.draw()
        plt.show()
    
    def remove_ticks_from_axis(self, ax):
        ax.xaxis.set_ticklabels([])
        # ax.xaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        # ax.yaxis.set_ticks([])
    
    def make_widgets(self):
        bw = .05
        bd = .1
        ts = .01
        width = .05
        top = .82
        axtext = self.fig.add_axes([ts, top, bw, width])
        axtext.set_axis_off()
        self.ci_text = axtext.text(0, .2, f'{self.current_index:{len(str(len(self.path_pairs)))}d}/{len(self.path_pairs)}', fontsize=11)
        current = top - (width + .01)
        axkeep = self.fig.add_axes([ts, current, bw, width])
        self.next_button = Button(ax=axkeep, label='next')
        self.next_button.on_clicked(self.next_press)
        current -= width + .01
        axremove = self.fig.add_axes([ts, current, bw, width])
        self.reset_button = Button(ax=axremove, label='reset')
        self.reset_button.on_clicked(self.reset_press)
        current -= width + .01
        axremove = self.fig.add_axes([ts, current, bw, width])
        self.back_button = Button(ax=axremove, label='back')
        self.back_button.on_clicked(self.back_press)

        axtext = self.fig.add_axes([.89, 0.74975-.025, bw, width])
        axtext.set_axis_off()
        self.standard_scores = axtext.text(0, 0, self.make_text(
            0, 0, 0, 0), fontsize=12, fontfamily='monospace'
                                           )

        axtext = self.fig.add_axes([.89, 0.26925-.025, bw, width])
        axtext.set_axis_off()
        self.ignored_scores = axtext.text(
            0, 0, self.make_text(0, 0, 0, 2), fontsize=12, fontfamily='monospace'
        )

    def add_stars(self, stat, num_stars):
        stars = '*' * num_stars
        return f'{stat}{stars}'

    def make_text(self, precision, recall, f1, num_stars=0):
        text = f'{self.add_stars("P", num_stars):>{2+num_stars}}: {precision:.2f}\n'
        text += f'{self.add_stars("R", num_stars):>{2+num_stars}}: {recall:.2f}\n'
        text += f'{self.add_stars("F1", num_stars):>{2+num_stars}}: {f1:.2f}'
        return text
    def make_plot(self):
        self.make_widgets()
        self.update_plots(init=True)
        # naip_data, sen_data, rows, cols = self.open_data()
        # # print(naip_data.max(), sen_data.max())
        # # print(sen_data.shape, naip_data.shape)
        # self.sen_im = self.ax[0].imshow(sen_data[rows.min():rows.max(), cols.min():cols.max()])
        # self.naip_im = self.ax[1].imshow(naip_data[rows.min():rows.max(), cols.min():cols.max()])
        # for i in [0, 1]:
        #     self.remove_ticks_from_axis(self.ax[i])
        # plt.subplots_adjust(top=0.98,bottom=0.08,left=0.005,right=0.72,hspace=0.09,wspace=0.0)
        # self.update_hists(naip_data, sen_data,rows, cols, first=True)


def name_to_shapely_box(file_name):
    box_info = [float(item) for item in file_name.split('/')[0].split('bbox_')[-1].split('_')]
    return shapely.box(*box_info)


def open_and_merge_dataframes(path_list):
    dfs = []
    for path in path_list:
        dfs.append(pd.read_parquet(path))
    df = pd.concat(dfs, ignore_index=True)
    return df


def get_stat_paths(base_path, model_index):
    stat_paths = [
        # base_path/f'evaluation_stats_{model_index}.parquet',
        # base_path/f'evaluation_test_{model_index}.parquet',
        base_path/f'evaluation_val_{model_index}.parquet'
    ]
    return stat_paths


def _add_data_to_dict(
        stat: str, stat_value: float, data_percent: float,
        model_number: int, stat_type: str, data_type: str, data_dict: defaultdict
):
    data_dict['stat_value'].append(stat_value)
    data_dict['stat'].append(stat)
    data_dict['model_number'].append(model_number)
    data_dict['stat_type'].append(stat_type)
    data_dict['data_type'].append(data_type)
    data_dict['percent'].append(data_percent)
    return data_dict

def add_to_data_dict(
        num_one_correct: int, num_model_one: int, num_target_one: int, num_correct: int, num_wrong: int,
        data_percent: float, model_number: int, stat_type: str, data_type: str, data_dict: defaultdict
):
    precision = num_one_correct / num_model_one
    recall = num_one_correct / num_target_one
    f1 = 2*precision*recall / (precision + recall)
    accuracy = num_correct / (num_correct + num_wrong)
    water_percent = num_target_one / (num_correct + num_wrong)
    data_dict = _add_data_to_dict(
        stat='precision', stat_value=precision, data_percent=data_percent,
        model_number=model_number, stat_type=stat_type, data_type=data_type, data_dict=data_dict
    )
    data_dict = _add_data_to_dict(
        stat='recall', stat_value=recall, data_percent=data_percent,
        model_number=model_number, stat_type=stat_type, data_type=data_type, data_dict=data_dict
    )
    data_dict = _add_data_to_dict(
        stat='accuracy', stat_value=accuracy, data_percent=data_percent,
        model_number=model_number, stat_type=stat_type, data_type=data_type, data_dict=data_dict
    )
    data_dict = _add_data_to_dict(
        stat='f1', stat_value=f1, data_percent=data_percent,
        model_number=model_number, stat_type=stat_type, data_type=data_type, data_dict=data_dict
    )
    data_dict = _add_data_to_dict(
        stat='water_percent', stat_value=water_percent, data_percent=data_percent,
        model_number=model_number, stat_type=stat_type, data_type=data_type, data_dict=data_dict
    )
    return data_dict


def compute_pixel_stats(
        df, data_percent:float, model_number: int, data_type: str,  data_dict=None,
):
    if data_dict is None:
        data_dict = defaultdict(list)
    if 'masked' not in df.columns:
        df['masked'] = 0
        df['masked_correct'] = 0
    df_agg = df[[
        'num_one_correct', 'num_model_one', 'num_target_one', 'num_correct',
        'num_wrong', 'num_fp_f', 'num_fn_f', 'masked', 'masked_correct'
    ]].sum()
    add_to_data_dict(
        num_one_correct=df_agg['num_one_correct'], num_model_one=df_agg['num_model_one'], data_percent=data_percent,
        num_target_one=df_agg['num_target_one'], num_correct=df_agg['num_correct'], num_wrong=df_agg['num_wrong'],
        model_number=model_number, stat_type='standard', data_type=data_type, data_dict=data_dict
    )
    add_to_data_dict(
        num_one_correct=df_agg['num_one_correct'] + df_agg['num_fp_f'],
        num_model_one=df_agg['num_model_one'],
        num_target_one=df_agg['num_target_one'] - df_agg['num_fn_f'] + df_agg['num_fp_f'],
        num_correct=df_agg['num_correct'] + df_agg['num_fn_f'] + df_agg['num_fp_f'],
        num_wrong=df_agg['num_wrong'] - df_agg['num_fn_f'] - df_agg['num_fp_f'], data_percent=data_percent,
        model_number=model_number, stat_type='model_correct', data_type=data_type, data_dict=data_dict
    )
    add_to_data_dict(
        num_one_correct=df_agg['num_one_correct'],
        num_model_one=df_agg['num_model_one'] - df_agg['num_fp_f'],
        num_target_one=df_agg['num_target_one'] - df_agg['num_fn_f'],
        num_correct=df_agg['num_correct'],
        num_wrong=df_agg['num_wrong'] - df_agg['num_fn_f'] - df_agg['num_fp_f'], data_percent=data_percent,
        model_number=model_number, stat_type='thick_ignored', data_type=data_type, data_dict=data_dict
    )
    add_to_data_dict(
        num_one_correct=df_agg['num_one_correct'] - df_agg['masked_correct'],
        num_model_one=df_agg['num_model_one'] - df_agg['masked_correct'],
        num_target_one=df_agg['num_target_one'] - df_agg['masked'],
        num_correct=df_agg['num_correct'] - df_agg['masked_correct'],
        num_wrong=df_agg['num_wrong'] - (df_agg['masked'] - df_agg['masked_correct']), data_percent=data_percent,
        model_number=model_number, stat_type='mask_ignored', data_type=data_type, data_dict=data_dict
    )
    return data_dict

def compute_pixel_stats_print(df):
    df_agg = df[[
        'num_one_correct', 'num_model_one', 'num_target_one',
        'num_correct', 'num_wrong', 'num_fp_f', 'num_fn_f'
    ]].sum()
    precision = df_agg['num_one_correct']/df_agg['num_model_one']
    precision_f = (df_agg['num_one_correct'] + df_agg['num_fp_f']) / df_agg['num_model_one']
    precision_e = df_agg['num_one_correct'] / (df_agg['num_model_one']-df_agg['num_fp_f'])

    recall = df_agg['num_one_correct']/df_agg['num_target_one']
    recall_f = (df_agg['num_one_correct'] + df_agg['num_fp_f'])/(df_agg['num_target_one'] - df_agg['num_fn_f'] + df_agg['num_fp_f'])
    recall_e = df_agg['num_one_correct']/(df_agg['num_target_one'] - df_agg['num_fn_f'])

    f1 = 2*precision*recall / (precision+recall)
    f1_f = 2*precision_f*recall_f / (precision_f+recall_f)
    f1_e = 2*precision_e*recall_e / (precision_e+recall_e)

    accuracy = (df_agg['num_correct']) / (df_agg['num_correct']+df_agg['num_wrong'])
    accuracy_f = (df_agg['num_correct'] + df_agg['num_fn_f'] + df_agg['num_fp_f']) / (df_agg['num_correct']+df_agg['num_wrong'])
    accuracy_e = (df_agg['num_correct']) / (df_agg['num_correct']+df_agg['num_wrong'] - df_agg['num_fn_f'] - df_agg['num_fp_f'])

    water_per = (df_agg['num_target_one']) / (df_agg['num_correct']+df_agg['num_wrong'])
    water_per_f = (df_agg['num_target_one'] - df_agg['num_fn_f'] + df_agg['num_fp_f']) / (df_agg['num_correct']+df_agg['num_wrong'])
    water_per_e = (df_agg['num_target_one']) / (df_agg['num_correct']+df_agg['num_wrong'] - df_agg['num_fp_f'] - df_agg['num_fn_f'])


    print(f'{"precision:":>12} {precision:.3f}, {"precision_f:":>12} {precision_f:.3f}, {"precision_e:":>12} {precision_e:.3f}')
    print(f'{"recall:":>12} {recall:.3f}, {"recall_f:":>12} {recall_f:.3f}, {"recall_e:":>12} {recall_e:.3f}')
    print(f'{"f1:":>12} {f1:.3f}, {"f1_f:":>12} {f1_f:.3f}, {"f1_e:":>12} {f1_e:.3f}')
    print(f'{"accuracy:":>12} {accuracy:.3f}, {"accuracy_f:":>12} {accuracy_f:.3f}, {"accuracy_e:":>12} {accuracy_e:.3f}')
    print(f'{"water_per:":>12} {water_per:.3f}, {"water_per_f:":>12} {water_per_f:.3f}, {"water_per_e:":>12} {water_per_e:.3f}')
    return precision_e, recall_e, f1_e, accuracy_e


def add_totals_to_df(df: pd.DataFrame, total_name: str,  types: list[str]):
    types_correct = [f'{type}_correct' for type in types]
    df[total_name] = df[types[0]]
    df[f'{total_name}_correct'] = df[types_correct[0]]
    for type, type_correct in zip(types[1:], types_correct[1:]):
        df[f'{total_name}_correct'] = df[f'{total_name}_correct'] + df[type_correct]
        df[total_name] = df[total_name] + df[type]
    return df


import pandas as pd


def df_row_to_latex_row(row: pd.Series, columns: list[str]) -> str:
    latex_row = "\t"
    for column in columns[:-1]:
        try:
            latex_row += f'{row[column]:.4f} & '
        except:
            latex_row += f'{make_column_entry(row[column])}' + ' & '
    latex_row += f'{100*row[columns[-1]]:.2f}\% \\\\\n'
    return latex_row

def make_column_entry(entry: str) -> str:
    new_entry = entry.replace("_", "\\\\").title()
    return '\\textbf{\\makecell{'+new_entry+'}}'


def make_column_row(columns: list[str]) -> str:
    latex_row = "\t"
    for column in columns[:-1]:
        latex_row += f'{make_column_entry(column)} & '
    latex_row += f'{make_column_entry(columns[-1])} \\\\\n'
    return latex_row


def df_to_latex_tabular(df: pd.DataFrame):
    num_rows = len(df)
    num_columns = len(df.columns)
    to_return = "\\begin{tabular}{|c|" + "|c"*(num_columns-1) + '|}\n'
    to_return += '\t\\hline\n'
    columns = df.columns.to_list()
    to_return += make_column_row(columns)
    to_return += '\t\\hline\n'

    for i in range(num_rows):
        to_return += df_row_to_latex_row(df.iloc[i], columns)
        to_return += '\t\\hline\n'
    to_return += '\\end{tabular}'
    return to_return



if __name__ == '__main__':
    import seaborn as sns
    from water.basic_functions import printdf
    inputs_832 = ppaths.training_data/'model_inputs_832'
    data_dict = defaultdict(list)
    # inputs_832 = ppaths.training_data/'model_inputs_224'
    # new_index = 825
    new_index = 841
    # new_index = 834
    # for new_index in [825, 831, 834, 836]:
    df_new = open_and_merge_dataframes(get_stat_paths(inputs_832, new_index))
    df_new = df_new.sort_values(by='file_name')
    df_new = df_new.set_index('file_name')
    df_832 = df_new
    df_832 = df_832.reset_index()
    # printdf(df_832)
    value_dict = {
        'target_val_0': 'land',
        'target_val_1': 'playa',
        'target_val_2': 'inundation',
        'target_val_3': 'swamp_i',
        'target_val_4': 'swamp_p',
        'target_val_5': 'swamp',
        'target_val_6': 'reservoir',
        'target_val_7': 'lake_i',
        'target_val_8': 'lake_p',
        'target_val_9': 'lake',
        'target_val_10': 'spillway',
        'target_val_11': 'drainage',
        'target_val_12': 'wash',
        'target_val_13': 'canal_storm',
        'target_val_14': 'canal_aqua',
        'target_val_15': 'canal',
        'target_val_16': 'artificial_path',
        'target_val_17': 'ephemeral',
        'target_val_18': 'intermittent',
        'target_val_19': 'perennial',
        'target_val_20': 'streams',
        'target_val_21': 'other',
        'target_val_22': 'intersections'
    }
    value_dict_2 = {f'{key}_correct': f'{value}_correct' for key, value in value_dict.items()}
    df_832 = df_832.rename(columns=value_dict)
    df_832 = df_832.rename(columns=value_dict_2)

    df_832 = add_totals_to_df(df_832, 'swamp_total', types=['swamp_i', 'swamp_p', 'swamp'])
    df_832 = add_totals_to_df(df_832, 'lake_total', types=['lake_i', 'lake_p', 'lake'])
    df_832 = add_totals_to_df(df_832, 'canal_total', types=['canal', 'canal_storm'])
    df_832 = add_totals_to_df(
        df_832, 'streams_total', types=['ephemeral', 'intermittent', 'perennial', 'streams']
    )
    types_to_mask = [
        'swamp_total', 'drainage', 'lake_i', 'playa',
        'inundation', 'spillway', 'wash', 'canal_total'
    ]
    df_832 = add_totals_to_df(df_832, 'masked', types_to_mask)
    df_832_all_removed = df_832[
        (df_832.swamp_total == 0) & (df_832.lake_i == 0) & (df_832.canal_total == 0) & (df_832.wash==0)
        & (df_832.inundation == 0) & (df_832.drainage==0) & (df_832.playa==0) & (df_832.spillway==0)
        ]
    # types = ['land', 'playa', 'inundation', 'swamp_i',
    #     'swamp_p', 'swamp', 'swamp_total', 'reservoir', 'lake_i', 'lake_p', 'lake', 'lake_total',
    #     'spillway', 'drainage', 'canal_aqua', 'canal', 'canal_total',
    #     'artificial_path', 'ephemeral', 'intermittent', 'perennial', 'streams', 'streams_total',
    #     'other', 'removed_from_training']
    types = ['swamp_total', 'lake_i', 'canal_total']

    print('Including all:')
    compute_pixel_stats_print(df_832)
    data_dict = compute_pixel_stats(
        df=df_832, data_percent=1, model_number=new_index, data_type=f'all_included', data_dict=data_dict
    )
    # data_dict = compute_pixel_stats(
    #     df=df_832_all_removed, data_percent=len(df_832_all_removed)/len(df_832), model_number=new_index,
    #     data_type='all_removed', data_dict=data_dict
    # )
    print('\n'*2)

    # for type in types:
    #     percent = len(df_832[df_832[type] == 0])/len(df_832)
    #     print(f'excluding {type}, count {len(df_832[df_832[type] == 0])} {len(df_832[df_832[type] == 0])/len(df_832):.2%}')
    #     df_type = df_832[df_832[type] == 0]
    #     pe, re, f1e, ae = compute_pixel_stats_print(df_type)
    #     data_dict = compute_pixel_stats(
    #         df=df_type, data_percent=percent, model_number=new_index, data_type=f'{type}_excluded', data_dict=data_dict
    #     )
    #     # print(df_type[['a_f', 'p_f', 'r_f', 'f1']].describe())
    #     print('')
    #     print(f'including {type}, count {len(df_832[df_832[type] > 0])} {len(df_832[df_832[type] > 0])/len(df_832):.2%}')
    #     percent = len(df_832[df_832[type] > 0])/len(df_832)
    #     df_type = df_832[df_832[type] > 0]
    #     pi, ri, f1i, ai = compute_pixel_stats_print(df_type)
    #     data_dict = compute_pixel_stats(
    #         df=df_type, data_percent=percent, model_number=new_index, data_type=f'{type}_included', data_dict=data_dict
    #     )
    #     # print(df_type[['a_f', 'p_f', 'r_f', 'f1']].describe())
    #     print('')
    #     print(f'Difference {type}')
    #     print(f'precision: {pe-pi:.3f}, recall: {re-ri:.3f}, f1: {f1e-f1i:.3f}, accuracy: {ae-ai:.3f}')
    #     print('\n'*2)
    # data_df = pd.DataFrame(data_dict)
    # data_df = data_df[['data_type', 'stat_type', 'stat', 'stat_value', 'percent']]
    # data_df = data_df[data_df.stat.isin(['precision', 'recall', 'f1'])]
    # data_df['stat'] = data_df['stat'].str.replace('precision', 'p').str.replace('recall', 'r')
    # data_df = data_df[data_df.stat_type.isin(['standard', 'thick_ignored', 'mask_ignored'])]
    # data_df['stat'] = data_df[['stat', 'stat_type']].apply(
    #     lambda row: f'{row.stat}*' if 'thick' in row.stat_type else f'{row.stat}**' if 'mask' in row.stat_type else row.stat, axis=1
    # )
    # pivot = data_df.pivot(columns='stat', index=['data_type'], values='stat_value')
    # pivot = pivot[[f'{stat}{"*"*n}' for n in range(3) for stat in ['p', 'r', 'f1']]].copy()
    # pivot['Data Percent'] = data_df.drop_duplicates('data_type', keep='first').set_index('data_type')['percent']
    # pivot_reset = pivot.reset_index()
    # data_types = ['All Data_Included', 'Data with any_mask type excluded'] + [f'data with_{col}' for col in pivot_reset.data_type.str.replace(
    #         'lake_i', 'intermittent lakes'
    #     ).str.replace('_total_', 's ')[2:]]
    # pivot_reset['data_type'] = data_types
    # pivot_reset.rename(columns={'data_type': 'Data Type'}, inplace=True)
    # # print(df_to_latex_tabular(pivot_reset))
    #
    # df_832['p_diff'] = df_832['p_f'] - df_832['p_n']
    # df_832['f1_diff'] = df_832['f1'] - df_832['f1_n']
    #
    # df_832.sort_values(by='f1_diff', ascending=False, inplace=True)
    # printdf(df_832)
    # df_plot = df_832
    # # df_plot = df_832[df_832['swamp_total'] > 69222]
    # eval_832 = [inputs_832/f'val_data/{file_name}' for file_name in df_plot.file_name]
    # chooser_832 = Chooser(
    #     eval_dir=inputs_832, eval_files=eval_832, model_number=new_index,
    #     output_dir=inputs_832/f'output_val_data_{new_index}', decrease_steps=1
    # )

    hulls_df = gpd.read_parquet(ppaths.training_data/'hu4_hulls.parquet')
    hulls_df = hulls_df.reset_index()
    # hulls_all = shapely.unary_union(hulls_df.geometry)
    df_832['geometry'] = df_832.file_name.apply(name_to_shapely_box)
    df_832 = gpd.GeoDataFrame(df_832, crs=4269)
    hu4_indices = [103, 204, 309, 403, 505, 601, 701, 805, 904, 1008, 1110, 1203, 1302, 1403, 1505, 1603, 1708, 1804]

    # for hu4_index in hu4_indices:
    #     hull = hulls_df[hulls_df.hu4_index == hu4_index].reset_index().geometry[0]
    #     # print('')
    #     # df_type = df_832[~df_832.intersects(hull)]
    #     # print(f'Excluding {hu4_index}, count {len(df_type)} {len(df_type)/len(df_832):.2%}')
    #     # percent = len(df_type)/len(df_832)
    #     # pe, re, f1e, ae = compute_pixel_stats_print(df_type)
    #     # data_dict = compute_pixel_stats(
    #     #     df=df_type, data_percent=percent, model_number=new_index,
    #     #     data_type=f'{hu4_index}_excluded', data_dict=data_dict
    #     # )
    #     # print(df_type[['a_f', 'p_f', 'r_f', 'f1']].describe())
    #     print('')
    #     df_type = df_832[df_832.intersects(hull)]
    #     print(f'including {hu4_index}, count {len(df_type)} {len(df_type)/len(df_832):.2%}')
    #     percent = len(df_type)/len(df_832)
    #     pi, ri, f1i, ai = compute_pixel_stats_print(df_type)
    #     data_dict = compute_pixel_stats(
    #         df=df_type, data_percent=percent, model_number=new_index,
    #         data_type=f'{hu4_index}_included', data_dict=data_dict
    #     )
    #     print('')
    #     print(pi, ri, f1i)
    #     # print(f'Difference {hu4_index}')
    #     # print(f'precision: {pe-pi:.3f}, recall: {re-ri:.3f}, f1: {f1e-f1i:.3f}, accuracy: {ae-ai:.3f}')
    #     print('\n'*2)
    # data_df = pd.DataFrame(data_dict)
    # data_df = data_df[['data_type', 'stat_type', 'stat', 'stat_value', 'percent']]
    # data_df = data_df[data_df.stat.isin(['precision', 'recall', 'f1'])]
    # data_df['stat'] = data_df['stat'].str.replace('precision', 'p').str.replace('recall', 'r')
    # data_df = data_df[data_df.stat_type.isin(['standard', 'thick_ignored', 'mask_ignored'])]
    # data_df['stat'] = data_df[['stat', 'stat_type']].apply(
    #     lambda row: f'{row.stat}*' if 'thick' in row.stat_type else f'{row.stat}**' if 'mask' in row.stat_type else row.stat, axis=1
    # )
    # pivot = data_df.pivot(columns='stat', index=['data_type'], values='stat_value')
    # pivot = pivot[[f'{stat}{"*"*n}' for n in range(3) for stat in ['p', 'r', 'f1']]].copy()
    # pivot['Data Percent'] = data_df.drop_duplicates('data_type', keep='first').set_index('data_type')['percent']
    # pivot_reset = pivot.reset_index()
    # data_types = [f'data with_{col.replace("_", " ")}' for col in pivot_reset.data_type[:-1]]+['All Data_Included']
    # pivot_reset['data_type'] = data_types
    # pivot_dfs = [pivot_reset[pivot_reset.data_type=='All Data_Included']]
    # pivot_dfs += [pivot_reset[pivot_reset.data_type==f'data with_{ind} included'] for ind in hu4_indices]
    # pivot_reset = pd.concat(pivot_dfs, ignore_index=True)
    # pivot_reset.rename(columns={'data_type': 'Data Type'}, inplace=True)
    # print(df_to_latex_tabular(pivot_reset))
    #
    df_832['p_diff'] = df_832['p_f'] - df_832['p_n']
    df_832['f1_diff'] = df_832['f1'] - df_832['f1_n']

    df_832.sort_values(by='f1_diff', ascending=False, inplace=True)
    printdf(df_832)
    df_plot = df_832[df_832.intersects(hulls_df[hulls_df.hu4_index==1603].reset_index().geometry[0])]
    # df_plot = df_832[df_832['swamp_total'] > 69222]
    eval_832 = [inputs_832/f'val_data/{file_name}' for file_name in df_plot.file_name]
    chooser_832 = Chooser(
        eval_dir=inputs_832, eval_files=eval_832, model_number=new_index,
        output_dir=inputs_832/f'output_val_data_{new_index}', decrease_steps=1
    )

    # df.to_parquet(ppaths.training_data/'model_inputs_832/val_shp.parquet')
    # bottom_f1 = df_832['f1'].quantile(.25)
    # top_f1 = df_832['f1'].quantile(.85)
    #
    # print(bottom_f1, top_f1)
    # bottom_p = df_832['p_f'].quantile(.25)
    # bottom_r = df_832['r_f'].quantile(.25)
    # df_832['keep'] = df_832[['file_name', 'f1']].apply(
    #     lambda x: (ppaths.training_data/f'model_inputs_832/input_data/{x.file_name}').exists() and x.f1>bottom_f1 and x.f1<top_f1
    #     , axis=1
    # )
    # df_832['keep'] = df_832[['file_name', 'f1', 'p_f', 'r_f']].apply(
    #     lambda x: (ppaths.training_data/f'model_inputs_832/input_data/{x.file_name}').exists() and x.f1>bottom_f1
    #     , axis=1
    # )
    # print(len(df_832))
    # df_keep_832 = df_832[df_832.keep]
    # print(len(df_keep_832))
    # printdf(df_keep_832[['file_name']])
    #
    # inputs_224 = ppaths.training_data/'model_inputs_224'
    # inputs_dir = inputs_224/'input_data'
    # file_names = []
    # for file_name in df_keep_832.file_name:
    #     file_parent = file_name.split('/')[0]
    #     if (inputs_dir/file_parent).exists():
    #         file_names.extend([f'{file_parent}/{file.name}' for file in (inputs_dir/file_parent).iterdir()])
    # print(len(file_names))
    # df_keep_224 = pd.DataFrame({'file_name': file_names})
    # df_keep_224.to_parquet(inputs_224/'new_inputs.parquet')
    # df_keep_832.to_parquet(inputs_832/'new_inputs.parquet')
    # tree = shapely.STRtree(df_832.geometry.to_numpy())
    # exterior = shapely.unary_union([geom.exterior for geom in hulls_all.geoms])
    # not_contained = tree.query(geometry=exterior, predicate='intersects')
    # df_832 = gpd.GeoDataFrame(df_832, crs=4326).reset_index(drop=True)
    # # ax = df_832.plot('f1', legend=True, vmin=.5, cmap='Greens')
    # ax = df_832.plot('f1', legend=True, vmin=0, vmax=1, cmap='Greens')
    # ax.set_title('f1')
    # ax.set_facecolor('black')
    # hulls_df.exterior.plot(ax=ax, color='black')
    # ax = df_832.plot('r_f', legend=True, vmin=0, vmax=1, cmap='Blues')
    # ax.set_title('r_f')
    # ax.set_facecolor('black')
    # hulls_df.exterior.plot(ax=ax, color='black')
    #
    # ax = df_832.plot('p_f', legend=True, vmin=0, vmax=1, cmap='Reds')
    # ax.set_title('p_f')
    # ax.set_facecolor('black')
    # hulls_df.exterior.plot(ax=ax, color='black')


    # df_832 = df_832.reset_index()
    # str_tree = shapely.STRtree(df_832.geometry.to_list())
    # df_832_idx_to_hull = {}
    # for idx, geom in zip(hulls_df.hu4_index, hulls_df.geometry):
    #     df_832_idxs = str_tree.query(geom, predicate='contains')
    #     for df_832_idx in df_832_idxs:
    #         df_832_idx_to_hull[df_832_idx] = idx
    # df_832['hu4_index'] = df_832['index'].apply(lambda x: df_832_idx_to_hull[x])
    #
    # # printdf(df_832.sort_values(by='f1')[['hu4_index', 'p_f', 'r_f', 'a_f', 'f1']], 20)
    #
    # df_sort = df_832[df_832.f1_diff > .2]
    # df_sort = df_sort.sort_values(by='hu4_index')

    # printdf(df_sort.describe(), 100)
    # df_sort = df_sort.sort_values(by='f1')
    # eval_832 = [inputs_832/f'input_data/{file_name}'
    #             for file_name in df_sort.file_name]
    # eval_832 = [inputs_832/f'input_data/{file_name}' for file_name in df_sort.file_name]
    # chooser_832 = Chooser(eval_dir=inputs_832, eval_files=eval_832, model_number=248)



    # # print(df_832.columns)
    # df_agg = df_832.groupby('hu4_index')[['p_mean', 'r_mean', 'f1_mean', 'a_mean']].mean()
    # df_agg['count'] = df_832.groupby('hu4_index')['p_mean'].count()
    # df_agg = gpd.GeoDataFrame(df_agg.join(hulls_df.set_index('hu4_index')), crs=4326)
    # df_agg.plot('f1_mean', legend=True, vmin=.5, vmax=.9)





    # printdf(df_agg.sort_values(by='f1'),10)
    # printdf(df_agg.sort_values(by='p_f'), 20)
    # printdf(df_agg.sort_values(by='a_f', ascending=False), 20)
    #
    # printdf(df_agg[['count', 'p_f', 'r_f', 'f1', 'a_f']].corr())
    # df_sort = df_832.sort_values(by='hu4_index')
    #



    # eval_832 = [inputs_832/f'input_data/{file_name}'
    #             for file_name in df_new[df_new.f1<.47].reset_index().file_name]
    # # eval_832 = [inputs_832/f'input_data/{file_name}' for file_name in df_new.file_name]
    # chooser_832 = Chooser(eval_dir=inputs_832, eval_files=eval_832, model_number=662)


    # df_sort = df_832[df_832.f1_mean<df_832.f1_mean.quantile(.05)]
    # df_sort = df_sort.sort_values(by='hu4_index')
    # eval_832 = [inputs_832/f'input_data/{file_name}'
    #             for file_name in df_sort.file_name]
    # chooser_832 = Chooser(eval_dir=inputs_832, eval_files=eval_832)


    # df_832[(df_832.r_f < .5) & (df_832.p_f < .5)].plot()
    # ax = df_832.plot('a_f', cmap='Greens', legend=True, vmin=.85)
    # ax.set_facecolor('black')
    #
    # split = .8
    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # df_832[df_832.f1 < split].plot(ax=ax[0])
    # df_832[df_832.f1 >= split].plot(ax=ax[1])
    # for axi in ax:
    #     axi.set_facecolor('black')
    # print(len(df_832[df_832.f1<split]), len(df_832[df_832.f1>=split]))
    #
    # split = .5
    # ax = df_832[df_832.f1 < split].plot(color='red', alpha=.75)
    # df_832[df_832.f1 >= split].plot(ax=ax, alpha=.75)
    # ax.set_facecolor('black')
    # # fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    # df_832.plot('f1', legend=False, cmap='Greens', ax=ax[0], vmin=.5)
    
    
    
    
    # df_832.plot('p_f', legend=False, cmap='Greens', ax=ax[0], vmin=.5)
    # df_832.plot('r_f', legend=False, cmap='Greens', ax=ax[1], vmin=.5)
    # for axi in ax:
    #     axi.set_facecolor('black')
    # df_832[['a_f', 'p_f', 'r_f', 'f1']].corr()
    # df_keep_224 = df_224[df_224.a_f >= df_224.a_f.quantile(.05)]
    # df_keep_224.to_parquet(inputs_224/'new_inputs.parquet')
    #
    # df_keep_416 = df_416[df_416.a_f >= df_416.a_f.quantile(.05)]
    # df_keep_416.to_parquet(inputs_416/'new_inputs.parquet')
    #
    # df_keep_832 = df_832[df_832.a_f >= df_832.a_f.quantile(.05)]
    # df_keep_832.to_parquet(inputs_832/'new_inputs.parquet')
    
    # eval_224 = [inputs_224/f'input_data/{file_name}'
    #             for file_name in df_224[df_224.a_f >= df_224.a_f.quantile(.05)].file_name]
    # chooser_224 = Chooser(eval_dir=inputs_224, eval_files=eval_224)
    #
    # eval_416 = [inputs_416/f'input_data/{file_name}'
    #             for file_name in df_416[df_416.a_f >= df_416.a_f.quantile(.05)].file_name]
    # chooser_416 = Chooser(eval_dir=inputs_416, eval_files=eval_416)
    #
    # eval_832 = [inputs_832/f'input_data/{file_name}'
    #             for file_name in df_832[df_832.f1 >= df_832.f1.quantile(.95)].file_name]
    # chooser_832 = Chooser(eval_dir=inputs_832, eval_files=eval_832)
    
    # print('224')
    # printdf(df_224.describe(.01*i for i in range(1, 10)), 100)
    # print('')
    # print('416')
    # printdf(df_416.describe(.01*i for i in range(1, 10)), 100)
    # print('')
    # print('832')
    # printdf(df_832.describe(.01*i for i in range(1, 10)), 100)

    # chooser = Chooser(eval_dir=ppaths.training_data/'model_inputs_832')
    
    # try:
    #     df
    # except:
    #     df = pd.read_csv(ppaths.training_data/'model_evaluation/evaluation_stats.csv')
    
    # print(df.columns)
    # to_keep = df[~((df.a_f < .5) & (df.p_f>.65)
    #                | ((df.p_f<.5) & (df.r_f<.5) & (df.a_f<.8))
    #                | ((df.a_f<.5) & (df.num_target_one<50) & (df.num_model_one>1000))
    #                | (df.num_target_one>1000) & (df.num_model_one<100))]
    # df_keep = df[~((df.a_f < .5) & (df.p_f>.65)
    #                | ((df.p_f<.5) & (df.r_f<.5) & (df.a_f<.8))
    #                | ((df.a_f<.5) & (df.num_target_one<50) & (df.num_model_one>1000))
    #                | (df.num_target_one>1000) & (df.num_model_one<100)
    #                | ((df.num_target_one==0) & (df.num_model_one>200)))]
    # files = df[((df.num_target_one==0) & (df.num_model_one>200))].file_name
    # df_keep = df[df.a_f>.86]
    # files = df[(df.a_f<.86)].file_name

    # printdf(df.describe(.01*i for i in range(1, 20)), 100)
    # # #
    # # #
    # file_paths = [ppaths.training_data/f'model_evaluation/input_data/{file}' for file in files]
    # chooser = Chooser(eval_files=file_paths)
    # print(len(df_keep), len(df), len(df) - len(df_keep))
    # df_keep = df[~(((df.recall<.5) & (df.precision<.5)) | ((df.accuracy<.75) & ((df.recall<.75) | (df.precision>.25)) ))]
    # df_keep = df_keep.reset_index(drop=True)
    # df_keep.to_parquet(ppaths.training_data/'model_evaluation/new_inputs.parquet')
    # sns.displot(data=df_keep, x='precision')
    # sns.displot(data=df_keep, x='recall')
    # try:
    #     df1, df2
    # except:
    #     df1 = pd.read_csv(ppaths.training_data/'model_evaluation/evaluation_stats.csv')
    #     df2 = pd.read_csv(ppaths.training_data/'model_evaluation/evaluation_stats_1.csv')
    #
    #     df1 = df1.sort_values(by='file_name').reset_index(drop=True)
    #     df2 = df2.sort_values(by='file_name').reset_index(drop=True)
    #
    # df2_keep = df2[~(
    #         (
    #                 (df2.precision<.5) & (df2.recall<.5)
    #         )
    #                 |
    #         (
    #                 (df2.accuracy<.75) & ((df2.recall<.75) | (df2.precision>.25))
    #         )
    # )]
    # df1_keep = df1.loc[df2_keep.index].reset_index(drop=True)
    # df1_keep1 = df1_keep[~(
    #         (
    #                 (df1_keep.f1 < .43)
    #         )
    #                 |
    #         (
    #                 (df1_keep.accuracy<.75) & ((df1_keep.recall<.75) | (df1_keep.precision>.25))
    #         )
    # )]
    # # print(len(df1))
    # df1_keep1.to_parquet(ppaths.training_data/'model_evaluation/new_inputs.parquet')
    # printdf(df1_keep1.describe(.05*i for i in range(1,20)),100)
    # print(len(df1_keep1), len(df2_keep))

    #
    # printdf(df1_keep1.describe(.05*i for i in range(1,20)),100)
    # print(len(df1_keep), len(df1_keep1))
    # df = df1_keep.copy()
    # df = df1_keep1
    # # printdf(df.describe(.05*i for i in range(1,20)),100)
    # Mval = .6
    # mval = .4
    # files = df.file_name
    #
    # # printdf(df[df.diffs<-.15], 100)
    # # #
    # file_paths = [ppaths.training_data/f'model_evaluation/input_data/{file}' for file in files]
    # chooser = Chooser(eval_files=file_paths)
    # chooser = Chooser()