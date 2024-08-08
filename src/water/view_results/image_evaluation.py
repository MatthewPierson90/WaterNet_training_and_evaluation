import matplotlib.pyplot as plt
import pandas as pd
import shapely
from matplotlib.widgets import TextBox, RadioButtons
import numpy as np
from water.basic_functions import ppaths
import torch
from torch import nn
import rasterio as rio
from matplotlib.widgets import Button, TextBox
from water.basic_functions import ppaths, tt, time_elapsed, open_json, save_json, my_pool, save_pickle, open_pickle
import rioxarray as rxr
from rasterio.enums import Resampling
from water.data_functions.clean.merge_waterway_data import ww_val_to_per
import geopandas as gpd
import time
from torch.nn import functional as F
import warnings
warnings.simplefilter('ignore', UserWarning)

class Chooser:
    def __init__(self, clear_seen=False):
        count = 0
        self.seen_path = ppaths.waterway/'model_evaluation/good_bad_ugly.json'
        self.num_seen = 0
        self.num_good_precision = 0
        self.num_good_recall = 0
        self.num_good_both = 0
        self.num_cloudy = 0
        if not self.seen_path.exists() or clear_seen:
            self.seen_names = {}
        else:
            self.seen_names = open_json(self.seen_path)
            self.num_seen = len(self.seen_names)
            self.count_current()
        eval_input_path = ppaths.waterway/'model_evaluation/input_data'
        eval_files = list(eval_input_path.glob('*'))
        # eval_files.sort(key=lambda x:x.stat().st_mtime, reverse=True)
        np.random.shuffle(eval_files)
        # time.sleep(5)
        self.waterway_file = None
        self.rsl=None
        self.sen_im=None
        path_pairs = []
        for eval_file in eval_files:
            tif_files = eval_file.glob('*.tif')
            for file in tif_files:
                path_pairs.append((file, eval_file/'waterways.parquet'))
        # path_pairs = [(eval_file, output_dir/eval_file.name) for eval_file in eval_files if (output_dir/eval_file.name).exists()]
        self.path_pairs = path_pairs
        self.num_remaining = len(self.path_pairs)
        # print(len(path_pairs))
        # np.random.shuffle(self.path_pairs)
        self.current_index = 0
        self.fig, self.ax = plt.subplots(1, 3, sharey=True, sharex=True)
        for i in range(3):
            self.ax[i].ticklabel_format(useOffset=False, style='plain')
        # for i in [0, 1]:
        #     self.remove_ticks_from_axis(self.ax[i])
        plt.subplots_adjust(top=0.995,
bottom=0.15,
left=0.035,
right=1.0,
hspace=0.07,
wspace=0.0)
        self.make_plot()


    def open_data(self):
        input_path, waterway_path = self.path_pairs[self.current_index]
        # with rio.open(output_path) as rio_f:
        #     output = np.round(rio_f.read()[0])
        with rxr.open_rasterio(input_path) as sen_ds:
            bbox = sen_ds.rio.bounds()
            print(sen_ds.shape)
            sen_data = sen_ds[1:4].to_numpy()
            sen_data = (sen_data.transpose((1,2,0))+1)/2
            gradient = sen_ds[-2].to_numpy()
            burned = sen_ds[-1].to_numpy()
            burned = ww_val_to_per(burned)
            # waterways = get_waterways_intersecting_bbox(bbox)
            waterways = gpd.read_parquet(waterway_path)
            waterways['geometry'] =waterways.geometry.buffer(.0001)
            return sen_data, gradient, burned, waterways, waterway_path, bbox

    def update_plots(self, init=False):
        if not init:
            self.current_index += 1
        self.ci_text.set_text(f'{self.current_index+self.num_seen:{len(str(self.num_remaining))}d}/{self.num_remaining}')
        sen_data, gradient, burned, waterways, ww_path, bbox = self.open_data()
        if burned.sum() < 10:
            self.next_press(1, 0, 0)
        else:
            self.bbox = bbox
            if self.sen_im is None:
                self.waterway_file = ww_path
                self.sen_im = self.ax[0].imshow(sen_data, extent=(bbox[0], bbox[2], bbox[1], bbox[3]))
                waterways.plot(ax=self.ax[0], alpha=.75, edgecolor='black',color='white')
                self.ax[0].set_aspect(1)
                self.naip_im = self.ax[1].imshow(sen_data, extent=(bbox[0], bbox[2], bbox[1], bbox[3]))
                self.old_mean = gradient.mean()
                self.old_std = gradient.std()
                self.out_im = self.ax[2].imshow(gradient, cmap='inferno', vmax=self.old_mean + self.old_std,
                                                vmin=0, extent=(bbox[0], bbox[2], bbox[1], bbox[3]))

            else:
                if self.waterway_file != ww_path:
                    self.waterway_file = ww_path
                    self.ax[0].clear()
                    self.sen_im = self.ax[0].imshow(sen_data, extent=(bbox[0], bbox[2], bbox[1], bbox[3]))
                    self.sen_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
                    self.ax[0] = waterways.plot(ax=self.ax[0], alpha=.75, edgecolor='black',color='white')
                    self.ax[0].set_aspect(1)
                else:
                    self.sen_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
                    self.sen_im.set_data(sen_data)
                self.naip_im.set_data(sen_data)
                self.naip_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
                self.out_im.set_data(gradient)
                self.out_im.set_extent((bbox[0], bbox[2], bbox[1], bbox[3]))
                self.old_mean = (self.current_index*self.old_mean + gradient.mean())/(self.current_index + 1)
                self.old_std = (self.current_index*self.old_std + gradient.std())/(self.current_index + 1)
                self.out_im.set_clim(vmin=0, vmax=gradient.mean()+gradient.std())
                self.reset_press(1)
            plt.locator_params(axis='y', nbins=5)
            plt.locator_params(axis='x', nbins=5)
            print(f'num good precision: {self.num_good_precision}')
            print(f'num good recall: {self.num_good_recall}')
            print(f'num good: {self.num_good_both}')
            print(f'num cloudy: {self.num_cloudy}')
            plt.draw()
            plt.show()

    def next_press(self, press, p_value, r_value):
        file_path, _ = self.path_pairs[self.current_index]
        key = f'{file_path.parent.name}/{file_path.name}'
        print(p_value, r_value)
        self.seen_names[key] = {'precision': p_value,
                                'recall': r_value}
        self._add_dict_to_totals(self.seen_names[key])
        save_json(self.seen_path, self.seen_names)
        self.update_plots()


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
        # self.ax[1].xaxis._set_lim(v0=left, v1=right, auto=True)
        # self.ax[1].yaxis._set_lim(v0=bottom, v1=top, auto=True)
        # self.ax[1].set_aspect(1)
        # self.ax[2].xaxis._set_lim(v0=left, v1=right, auto=True)
        # self.ax[2].yaxis._set_lim(v0=bottom, v1=top, auto=True)
        # self.ax[2].set_aspect(1)
        # self.ax[1].set_position((left, bottom, width, height))
        plt.draw()
        plt.show()

    # def rotate_ticks(self):
    #     for ax in self.ax:
    #         ax.xticks(rotation=45, ha='right')

    def remove_ticks_from_axis(self, ax):
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])


    def make_widgets(self):
        bw = .08
        bd = .1
        ts = .0625
        axtext = self.fig.add_axes([ts+.01, .02, bw, .05])
        # self.remove_ticks_from_axis(axtext)
        axtext.set_axis_off()
        self.ci_text = axtext.text(0,.3, f'{self.current_index+self.num_seen:{len(str(self.num_remaining))}d}/{self.num_remaining}', fontsize=14)
        # TextBox(ax=axtext,
        #         label_pad=0,
        #         label=f'',
        #         initial=f'{self.current_index}/{len(self.path_pairs)}',
        #         textalignment='right')
        current = ts + .01*(len(self.ci_text._text))
        axkeep = self.fig.add_axes([current, .02, bw, .05])
        self.good_button = Button(ax=axkeep, label='good')
        self.good_button.on_clicked(lambda x: self.next_press(x, 1, 1))
        current += bw + .01
        axkeep = self.fig.add_axes([current, .02, bw, .05])
        self.gp_button = Button(ax=axkeep, label='good precision')
        self.gp_button.on_clicked(lambda x: self.next_press(x, 1, 0))
        current += bw + .01
        axkeep = self.fig.add_axes([current, .02, bw, .05])
        self.gr_button = Button(ax=axkeep, label='good recall')
        self.gr_button.on_clicked(lambda x: self.next_press(x, 0, 1))
        current += bw + .01
        axkeep = self.fig.add_axes([current, .02, bw, .05])
        self.bad_button = Button(ax=axkeep, label='bad')
        self.bad_button.on_clicked(lambda x: self.next_press(x, 0, 0))
        current += bw + .01
        axkeep = self.fig.add_axes([current, .02, bw, .05])
        self.clouds = Button(ax=axkeep, label='clouds')
        self.clouds.on_clicked(lambda x: self.next_press(x, -1, -1))
        current += bw + .01
        axremove = self.fig.add_axes([current, .02, bw, .05])
        self.reset_button = Button(ax=axremove, label='reset')
        self.reset_button.on_clicked(self.reset_press)
        current += bw + .01
        axremove = self.fig.add_axes([current, .02, bw, .05])
        self.back_button = Button(ax=axremove, label='back')
        self.back_button.on_clicked(self.back_press)


    def _add_dict_to_totals(self, pr_dict):
        self.num_good_precision += max(pr_dict['precision'], 0)
        self.num_good_recall += max(pr_dict['recall'], 0)
        if pr_dict['precision'] == 1 and pr_dict['recall'] == 1:
            self.num_good_both += 1
        if pr_dict['precision'] == -1:
            self.num_cloudy += 1

    def count_current(self):
        for pr_dict in self.seen_names.values():
            self._add_dict_to_totals(pr_dict)

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


if __name__ == '__main__':
    chooser = Chooser(False)