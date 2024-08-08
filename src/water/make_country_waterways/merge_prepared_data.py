import warnings

import rasterio as rio
import numpy as np
from water.basic_functions import ppaths
import matplotlib.pyplot as plt


def make_slope(el_data):
    el_data = el_data[0]
    x_der = np.zeros(el_data.shape)
    y_der = np.zeros(el_data.shape)
    x_del = (el_data[:, 1:] - el_data[:, :-1])
    x_der[:, 1:-1] = (x_del[:, :-1] + x_del[:, 1:])/2
    x_der[:, 0] = x_del[:, 0]
    x_der[:, -1] = x_del[:, -1]
    
    y_del = (el_data[:-1] - el_data[1:])
    y_der[1:-1] = (y_del[:-1] + y_del[1:])/2
    y_der[0] = y_del[0]
    y_der[-1] = y_del[-1]
    
    x_der = np.stack([x_der])
    y_der = np.stack([y_der])
    return x_der, y_der


def make_veg_indices(sen_data):
    ndwi = (sen_data[2:3] - sen_data[0:1])/(sen_data[2:3] + sen_data[0:1] + .000000001)
    ndvi = (sen_data[0:1] - sen_data[1:2])/(sen_data[0:1] + sen_data[1:2] + .000000001)
    sen_data1 = np.concatenate([sen_data, ndvi, ndwi], axis=0)
    # print(sen_data.shape, sen_data1.shape)
    return sen_data1


def open_single_sen_image(sentinel_path, make_indices=False):
    with rio.open(sentinel_path) as sen_f:
        sen_data = sen_f.read()
        sen_data = sen_data.astype(np.float32)/255
        sen_missing = sen_data.sum(axis=0)
        sen_missing[sen_missing > 0] = 1
        if make_indices:
            sen_data = make_veg_indices(sen_data)
        sen_data = 2*sen_data - 1
    return sen_data, sen_missing


def open_elevation_data(elevation_path):
    with rio.open(elevation_path) as el_f:
        el_data = el_f.read().astype(np.float32)
        el_missing = el_data[0].copy()
        el_missing[(el_missing == np.nan) | (el_missing <= -999999) | (el_missing >= 10000)] = -999999
        el_missing[el_missing != -999999.0] = 1
        el_missing[(el_missing == -999999.0)] = 0
        el_missing = el_missing.astype(bool)
        el_data_min = el_data[np.stack([el_missing], axis=0)].min()
        el_data = el_data.copy() - el_data_min
        x_der, y_der = make_slope(el_data)
        xy_grad = np.sqrt(x_der**2 + y_der**2)
        grad = np.concatenate([x_der, y_der, xy_grad])
    return el_data, grad, el_missing


def make_placeholder_data(shape):
    placeholder_data = np.zeros(shape)
    placeholder_missing = np.zeros((shape[-2], shape[-1]))
    return placeholder_data, placeholder_missing


def open_all_and_merge(sentinel_path,
                       elevation_path,
                       make_indices=False,
                       fill_missing=False,
                       shape=(1, 832, 832)):
    sen_exists = sentinel_path.exists()
    elevation_exists = elevation_path.exists()
    make_placeholder_sentinel_data = False
    make_placeholder_elevation_data = False
    if not sentinel_path.exists():
        # print(f'sentinel path exists: {sentinel_path.exists()}\n\t{sentinel_path}')
        make_placeholder_sentinel_data = True
    if not elevation_path.exists():
        # print(f'elevation path exists: {elevation_path.exists()}\n\t{elevation_path}')
        make_placeholder_elevation_data = True
    if not make_placeholder_sentinel_data:
        sen_data, sen_missing = open_single_sen_image(sentinel_path, make_indices=make_indices)
    if not make_placeholder_elevation_data:
        el_data, grad, el_missing = open_elevation_data(elevation_path)
    if make_placeholder_sentinel_data:
        placeshape = [6 if make_indices else 4, shape[-2], shape[-1]]
        sen_data, sen_missing = make_placeholder_data(placeshape)
    if make_placeholder_elevation_data:
        el_placeshape = [1, shape[-2], shape[-1]]
        grad_placeshape = [3, shape[-2], shape[-1]]
        el_data, el_missing = make_placeholder_data(el_placeshape)
        grad, _ = make_placeholder_data(grad_placeshape)
    rows, cols = np.where(sen_missing == 0)
    sen_data[:, rows, cols] = np.nan
    rows, cols = np.where(el_missing == 0)
    el_data[:, rows, cols] = np.nan
    grad[:, rows, cols] = np.nan
    data = np.concatenate([sen_data, el_data, grad], axis=0, dtype=np.float32)
    return data, sen_missing, el_missing


if __name__ == '__main__':
    cnt = 'rwanda'
    files = (ppaths.waterway/f'country_data/{cnt}/sentinel_4326_cut').glob('*')
    files = list(files)
    file = files[1]
    d, m1, m2 = open_all_and_merge(file, cnt)
    # fig, ax = plt.subplots()
    # ax.imshow(m2)
