import rasterio as rio
import numpy as np
from water.basic_functions import ppaths
import matplotlib.pyplot as plt


def make_slope(el_data):
    el_data = el_data[0]
    x_der = np.zeros(el_data.shape)
    y_der = np.zeros(el_data.shape)
    x_del = (el_data[:, 1:] - el_data[:, :-1])
    x_der[:,1:-1] = (x_del[:, :-1] + x_del[:, 1:])/2
    x_der[:, 0] = x_del[:, 0]
    x_der[:, -1] = x_del[:, -1]

    y_del = (el_data[:-1] - el_data[1:])
    y_der[1:-1] = (y_del[:-1] + y_del[1:])/2
    y_der[0] = y_del[0]
    y_der[-1] = y_del[-1]

    # grad = (x_der**2 + y_der**2)**.5
    x_der = np.stack([x_der])
    y_der = np.stack([y_der])
    return x_der, y_der

def make_veg_indices(sen_data):
    ndwi = (sen_data[2:3] - sen_data[0:1]) / (sen_data[2:3] + sen_data[0:1] + .000000001)
    ndvi = (sen_data[0:1] - sen_data[1:2]) / (sen_data[0:1] + sen_data[1:2] + .000000001)
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

def open_sen_data(sentinel_path, is_time_series, make_indices=False):
    if is_time_series:
        sen_data_list = []
        sen_missing_all = None
        for i in range(3):
            sen_data, sen_missing = open_single_sen_image(sentinel_path/f'{i}.tif', make_indices=make_indices)
            sen_data_list.append(sen_data)
            if sen_missing_all is None:
                sen_missing_all = sen_missing
            else:
                sen_missing_all *= sen_missing
        sen_data_all = np.concatenate(sen_data_list, axis=0, dtype=np.float32)
        return sen_data_all, sen_missing_all
    else:
        return open_single_sen_image(sentinel_path, make_indices=make_indices)


def open_all_and_merge(file_index=0,
                       burned_path=None,
                       use_ts=True,
                       elevation_dir_name='elevation_cut',
                       burned_dir_path=None,
                       sentinel_dir_path=None,
                       make_indices=False,
                       fill_missing=False):
    ts = ''
    if use_ts:
        ts = '_ts'
    if burned_dir_path is None:
        burned_dir_path = ppaths.waterway/f'waterways{ts}_burned'
    burned_paths = list((burned_dir_path).glob('*'))
    burned_paths.sort(key=lambda x:x.name)
    if burned_path is None:
        burned_path = burned_paths[file_index]
    file_name = burned_path.name
    file_sub_name = f'{file_name}'
    sen_sub_name = file_sub_name
    if use_ts:
        sen_sub_name = sen_sub_name.split('.tif')[0]
    if sentinel_dir_path is None:
        sentinel_dir_path = ppaths.waterway/f'sentinel{ts}_4326'
    sentinel_path = sentinel_dir_path/ f'{sen_sub_name}'
    elevation_path = ppaths.waterway/f'{elevation_dir_name}/{file_sub_name}'
    if not sentinel_path.exists() or not elevation_path.exists() or not burned_path.exists():
        print(sentinel_path.exists(), elevation_path.exists(), burned_path.exists())
        return None, None
    else:
        sen_data, sen_missing = open_sen_data(sentinel_path, is_time_series=use_ts, make_indices=make_indices)
        with rio.open(elevation_path) as el_f:
            el_data = el_f.read().astype(np.float32)
            el_missing = el_data[0].copy()
            el_missing[(el_missing == np.nan) | (el_missing <= -999999) | (el_missing >= 10000)] = -999999
            el_missing[el_missing != -999999.0] = 1
            el_missing[(el_missing == -999999.0)] = 0
            el_missing = el_missing.astype(bool)
            if np.any(el_missing):
                el_data_min = el_data[np.stack([el_missing], axis=0)].min()
                el_data = el_data.copy()-el_data_min
                x_der, y_der = make_slope(el_data)
                # xx_der, xy_der = make_slope(x_der)
                # yx_der, yy_der = make_slope(y_der)
                xy_grad = np.sqrt(x_der**2 + y_der**2)
                grad = np.concatenate([x_der, y_der, xy_grad])
                # grad = np.concatenate([x_der, y_der, xx_der, xy_der, yx_der, yy_der, xy_grad])
            else:
                grad = np.zeros((4, el_data.shape[1], el_data.shape[2]), dtype=np.float32)
        with rio.open(burned_path) as ww_f:
            ww_data = ww_f.read().astype(np.float32)
            water = ww_data[-1:].copy()
            waterways = ww_data[0].copy()
            # print(len(waterways[waterways>0]))
            water[(water < 50) & (waterways == 0)] = 0
        # try:
            # missing = sen_missing*el_missing
            rows, cols = np.where(sen_missing == 0)
            sen_data[:, rows, cols] = np.nan
            rows, cols = np.where(el_missing == 0)
            el_data[:, rows, cols] = np.nan
            grad[:, rows, cols] = np.nan
            # print(sen_data.shape, el_data.shape, grad.shape, water.shape)
            data = np.concatenate([sen_data, el_data, grad, water], axis=0, dtype=np.float32)
            # data[:-1, r, c] = np.inf
        # except:
        #     return None, None
        # data = data.astype(np.float16)
    return data, waterways

def conv_mean(data:np.ndarray, row, col, side_len):
    rm = max(0, row-side_len)
    rM = min(data.shape[-1], row+side_len+1)
    cm = max(0, col-side_len)
    cM = min(data.shape[-1], col+side_len+1)
    data[:,row, col] = np.nanmean(data[:,rm:rM, cm:cM], axis=(1,2), keepdims=False, dtype=np.float32)
    return data



def ww_val_to_per(data):
    data[data==10] = .1
    data[data==20] = .25
    data[data==30] = .4
    data[data==40] = .6
    data[data==50] = 1.
    return data

def cut_data(data,
             ww_data,
             num_per_image=None,
             num_pixels=100,
             step_size=50,
             no_water_per=.02,
             max_water_per=.5):
    _, num_rows, num_cols = data.shape
    num_water = 0
    num_no_water = 0
    num_too_much_water = 0
    total = 0
    data_list = []
    rows = list(range(0, num_rows-num_pixels, step_size))
    cols = list(range(0, num_cols-num_pixels, step_size))
    if num_per_image is None:
        num_per_image = 10000000
    else:
        np.random.shuffle(rows)
        np.random.shuffle(cols)
    broke = False
    for row in rows:
        if broke:
            break
        for col in cols:
            if len(data_list) >= num_per_image:
                broke = True
                break
            total+=1
            chunk = data[:, row:row+num_pixels, col:col+num_pixels].copy()
            ww_chunk = ww_data[row:row+num_pixels, col:col+num_pixels]
            if np.any(np.isnan(chunk)):
                continue
            # chunk[4] = chunk[4] - chunk[4].min()
            # print(chunk[4].min(), chunk[4].max())
            if chunk[4].max()<0:
                continue
            watercells = np.where(ww_chunk > 0)[0]
            num_watercells = len(watercells)
            water_per = num_watercells/num_pixels**2
            # print(water_per)
            if water_per > max_water_per:
                num_too_much_water += 1
            elif water_per > no_water_per:
                chunk[-1] = ww_val_to_per(chunk[-1].copy())
                data_list.append(chunk)
                num_water += 1
            elif num_water/8 >= num_no_water:
                chunk[-1] = ww_val_to_per(chunk[-1].copy())
                data_list.append(chunk)
                num_no_water += 1

    # if len(data_list)>0:
    #     data_list = np.stack(data_list, axis=0)
    return data_list
    # else:
    #     return np.array([])



if __name__ == '__main__':
    d, w = open_all_and_merge(1, use_ts=False, fill_missing=True, elevation_dir_name='elevation_cap_cut')
    # dl = cut_data(d,w, 100, 50)

    # for d in dl:
    #     if dl[0, 4].max()< 0:
    #         raise Exception
    # plt.imshow(m)
    # print(d.shape)