import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import rasterio as rio
from matplotlib.widgets import Button
from water.basic_functions import ppaths, open_json, save_json
import rioxarray as rxr
from water.data_functions.clean.merge_waterway_data import ww_val_to_per
from water.data_functions.load.load_waterway_data_v3 import SenElBurnedLoader, SenBurnedLoader
from water.training.evaluate_model import get_input_files
import geopandas as gpd
from pathlib import Path
import shapely
from torch.nn import functional as F


def decrease_y(y, num_factors):
    y = torch.tensor(np.stack([y]))
    for i in range(num_factors):
        y = F.max_pool2d(y, 2)
    y = y.numpy()[0]
    return y


def open_hu4_parquet_data(index):
    return gpd.read_parquet(ppaths.waterway/f'hu4_parquet/hu4_{index:04d}.parquet')


def make_input_output_files(base_dir: Path, input_files: list[Path], model_number: int):
    output_files = [base_dir/f'output_data_{model_number}'/file.parent.name/file.name for file in input_files]
    return list(zip(input_files, output_files))


class Chooser:
    def __init__(self, model_number, clear_seen=False,
                 eval_files=None, path_pairs=None,
                 eval_dir=ppaths.waterway/'model_inputs_224',
                 data_loader: SenBurnedLoader = SenBurnedLoader()
                 ):
        count = 0
        self.data_loader = data_loader
        self.eval_dir = eval_dir
        self.seen_path = eval_dir/f'good_bad_ugly_model_eval.json'
        self.hu4_hulls = gpd.read_parquet(ppaths.waterway/'hu4_hulls.parquet')
        self.hull_tree = shapely.STRtree(self.hu4_hulls.geometry.to_list())
        self.index_to_hu4_index = {ind: idx for ind, idx in enumerate(self.hu4_hulls.hu4_index)}
        self.current_hu4 = 0
        self.hu4_gdf = open_hu4_parquet_data(101)
        if not self.seen_path.exists() or clear_seen:
            self.seen_names = {}
        else:
            self.seen_names = open_json(self.seen_path)
        if eval_files is None:
            eval_files = get_input_files(eval_dir=eval_dir)
        self.rsl = None
        self.sen_im = None
        output_dir = ppaths.waterway/f'{eval_dir}/output_data_{model_number}'
        self.path_pairs = path_pairs
        if path_pairs is None:
            self.path_pairs = make_input_output_files(self.eval_dir, eval_files, model_number)
        print(len(self.path_pairs))
        # np.random.shuffle(self.path_pairs)
        self.current_index = 0
        self.fig, self.ax = plt.subplots(1, 3, sharex=True, sharey=True)
        for i in range(3):
            self.ax[i].ticklabel_format(useOffset=False, style='plain')
            self.ax[i].get_xaxis().get_major_formatter().set_scientific(False)
        # for i in [0, 1]:
        #     self.remove_ticks_from_axis(self.ax[i])
        plt.subplots_adjust(
            top=0.995, bottom=0.15, left=0.035, right=1.0, hspace=0.07, wspace=0.0
        )
        self.make_plot()

    def open_data(self):
        input_path, output_path = self.path_pairs[self.current_index]
        # print(f'{input_path.parent.name}/{input_path.name}')
        # print(f'{output_path.parent.name}/{output_path.name}')
        with rio.open(output_path) as rio_f:
            output = np.round(rio_f.read()[0] + .2)
            # output = rio_f.read()[0]

            bbox = tuple(rio_f.bounds)

        sen_ds = self.data_loader.open_data(input_path)
        sen_data = sen_ds[1:4]
        sen_data = (sen_data.transpose((1, 2, 0)) + 1)/2
        gradient = sen_ds[-2]
        burned = sen_ds[-1]
        burned = decrease_y(burned, 2)
        burned[burned > 1] = 1
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
        self.burned = burned
        self.diff = output - burned
        self.bbox = bbox
        shapely_box = shapely.box(*bbox)
        # hu4_index = self.index_to_hu4_index[self.hull_tree.query(shapely_box, predicate='covered_by')[0]]
        # if hu4_index != self.current_hu4:
        #     self.current_hu4 = hu4_index
        #     self.hu4_gdf = open_hu4_parquet_data(self.current_hu4)
        # intersects_box = self.hu4_gdf[self.hu4_gdf.intersects(shapely_box)]
        # print(intersects_box.fcode_description.unique())
        # p, r, f1, a = self.print_image_stats()
        if self.sen_im is None:
            self.sen_im = self.ax[0].imshow(sen_data, extent=(bbox[0], bbox[2], bbox[1], bbox[3]))
            # self.plot_ww(ax=self.ax[0], intersects_df=intersects_box)
            self.ax[0].set_aspect(1)
            self.naip_im = self.ax[1].imshow(
                burned, vmax=1, vmin=0, extent=(bbox[0], bbox[2], bbox[1], bbox[3])
            )
            self.out_im = self.ax[2].imshow(
                output, vmax=.75, vmin=.25, extent=(bbox[0], bbox[2], bbox[1], bbox[3])
            )
        else:
            self.ax[0].clear()
            self.sen_im = self.ax[0].imshow(sen_data, extent=(bbox[0], bbox[2], bbox[1], bbox[3]))
            self.sen_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
            # self.ax[0] = ww.plot(ax=self.ax[0], alpha=.5)
            # self.plot_ww(ax=self.ax[0], intersects_df=intersects_box)
            self.ax[0].set_aspect(1)
            self.naip_im.set_data(burned)
            self.naip_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
            self.out_im.set_data(output)
            self.out_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
        for i in range(2):
            self.ax[i].ticklabel_format(useOffset=False, style='plain')
            self.ax[i].get_xaxis().get_major_formatter().set_scientific(False)
        plt.draw()
        plt.show()

    def plot_ww_single(self, ax, df, color):
        if len(df) > 0:
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
                self.diff[row, col] = 0
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
        fp_w1, total_fp = self.get_within_n_single(cur_model, cur_burned, 1, 1)
        fn_w1, total_fn = self.get_within_n_single(cur_model, cur_burned, 0, 1)
        if total_fp == 0:
            total_fp = 1
        if total_fn == 0:
            total_fn = 1
        accuracy = len(cur_diff[cur_diff == 0])/(cur_diff.shape[-1]*cur_diff.shape[-2])

        print(f' accuracy: {accuracy:.2%}')
        p = num_one_correct/num_model_one
        r = num_one_correct/num_burned_one
        f1 = 2*p*r/(p + r) if p + r > 0 else 0
        print(f'precision: {p: .2%}')
        print(f'   recall: {r: .2%}')
        print(f'       f1: {f1: .2%}')
        p = (num_one_correct + fp_w1)/num_model_one
        r = (num_one_correct)/(max(num_burned_one - fn_w1, 1))
        f1 = 2*p*r/(p + r) if p + r > 0 else 0
        print(f'precision1: {p:.2%}')
        print(f'   recall1: {r:.2%}')
        print(f'       f11: {f1:.2%}')

        print(f'per fp w1: {fp_w1/total_fp:.2%} ({fp_w1}/{total_fp})')
        print(f'per fn w1: {fn_w1/total_fn:.2%} ({fn_w1}/{total_fn})')
        print('')
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
        self.ax[0].xaxis._set_lim(v0=left, v1=right, auto=True)
        self.ax[0].yaxis._set_lim(v0=bottom, v1=top, auto=True)
        self.ax[0].set_aspect(1)
        self.ax[1].xaxis._set_lim(v0=left, v1=right, auto=True)
        self.ax[1].yaxis._set_lim(v0=bottom, v1=top, auto=True)
        self.ax[1].set_aspect(1)
        self.ax[2].xaxis._set_lim(v0=left, v1=right, auto=True)
        self.ax[2].yaxis._set_lim(v0=bottom, v1=top, auto=True)
        self.ax[2].set_aspect(1)
        # self.ax[1].set_position((left, bottom, width, height))
        plt.draw()
        plt.show()

    def remove_ticks_from_axis(self, ax):
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])

    def make_widgets(self):
        bw = .08
        bd = .1
        ts = .0625
        axtext = self.fig.add_axes([ts + .01, .02, bw, .05])
        axtext.set_axis_off()
        self.ci_text = axtext.text(0, .3,
                                   f'{self.current_index:{len(str(len(self.path_pairs)))}d}/{len(self.path_pairs)}',
                                   fontsize=14)
        current = ts + .01*(len(self.ci_text._text))
        axkeep = self.fig.add_axes([current, .02, bw, .05])
        self.next_button = Button(ax=axkeep, label='next')
        self.next_button.on_clicked(self.next_press)
        current += bw + .01
        axremove = self.fig.add_axes([current, .02, bw, .05])
        self.reset_button = Button(ax=axremove, label='reset')
        self.reset_button.on_clicked(self.reset_press)
        current += bw + .01
        axremove = self.fig.add_axes([current, .02, bw, .05])
        self.back_button = Button(ax=axremove, label='back')
        self.back_button.on_clicked(self.back_press)

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


if __name__ == '__main__':
    import seaborn as sns
    from water.basic_functions import printdf

    inputs_832 = ppaths.waterway/'model_inputs_832'
    index = 584
    df_832 = pd.read_parquet(inputs_832/f'evaluation_stats_{index}.parquet')
    df_832 = df_832.reset_index()
    value_dict = {
        'target_val_0': 'land',
        'target_val_1': 'trail',
        'target_val_2': 'road',
    }
    df_832 = df_832.rename(columns=value_dict)
    print(df_832.columns)
    printdf(df_832.describe(.05*i for i in range(1, 20)), 100)

    df = df_832[df_832.trail>0]
    print(len(df_832), len(df))
    df.to_parquet(inputs_832/'new_inputs.parquet')

    # df_keep = pd.DataFrame({'file_names': files})

    printdf(df_832[[
        'a_f', 'p_f', 'r_f', 'f1', 'land', 'trail', 'road',
    ]].corr()[['a_f', 'p_f', 'r_f', 'f1']], 100)
    # # #
    # printdf(df_832[['a_f', 'p_f', 'r_f', 'f1']].describe(.05*i for i in range(1, 20)), 100)
    hulls_df = gpd.read_parquet(ppaths.waterway/'hu4_hulls.parquet')
    df_832['geometry'] = df_832.file_name.apply(name_to_shapely_box)
    df_832 = gpd.GeoDataFrame(df_832, crs=4326).reset_index(drop=True)
    ax = df_832.plot('f1', legend=True, vmin=0, vmax=1, cmap='Greens')
    ax.set_title('f1')
    ax.set_facecolor('black')
    ax = df_832.plot('r_f', legend=True, vmin=0, vmax=1, cmap='Greens')
    ax.set_title('r_f')
    ax.set_facecolor('black')
    ax = df_832.plot('p_f', legend=True, vmin=0, vmax=1, cmap='Greens')
    ax.set_title('p_f')
    ax.set_facecolor('black')
    ax = df_832.plot('trail', legend=True, vmin=0, vmax=1, cmap='Greens')
    ax.set_title('p_f')
    ax.set_facecolor('black')