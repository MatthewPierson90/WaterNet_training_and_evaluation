import os

import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import rasterio as rio
from rasterio import transform
import shapely
from pyproj import Geod
import numpy as np
from water.basic_functions import ppaths, get_country_polygon, delete_directory_contents


def make_grid(grid_width, min_lon, max_lon, min_lat, max_lat, crs, save_dir):
    file_name = f'bbox_{min_lon:.10f}_{min_lat:.10f}_{max_lon:.10f}_{max_lat:.10f}.tif'
    save_path = save_dir/file_name
    trans = transform.from_bounds(min_lon, min_lat, max_lon, max_lat, grid_width, grid_width)
    # print(trans)
    # print(min_lon, min_lat, max_lon, max_lat)
    # print('')
    data = np.ones((1, grid_width, grid_width))
    meta = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': 0,
        'width': grid_width,
        'height': grid_width,
        'count': 1,
        'crs': crs,
        'transform': trans
    }
    with rio.open(save_path, 'w', **meta) as dst_f:
        dst_f.write(data)

def make_grids_for_fixed_lats(min_lat, max_lat, west, east, x_res, grid_width,
                              step_size, save_dir, polygon, crs, geod, grid_res):
    # mid_lat = (max_lat + min_lat)/2
    # new_lon, _, _ = geod.fwd(lons=west, lats=mid_lat, dist=grid_res, radians=False, az=90)
    # x_res = abs(west - new_lon)
    # west -= x_res*grid_width/2
    # east += x_res*grid_width/2

    width = grid_width*x_res
    total_cols = int(np.ceil((east - west)/x_res))
    num_steps = total_cols//step_size + 1
    for i in range(num_steps):
        if east == 179.999:
            max_lon = east - step_size*i*x_res
            min_lon = max_lon - width
        else:
            min_lon = west + step_size*i*x_res
            max_lon = min(min_lon + width, 179.999)
        if polygon.intersects(shapely.box(min_lon, min_lat, max_lon, max_lat)):
            make_grid(
                min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,
                grid_width=grid_width, crs=crs, save_dir=save_dir
            )



def make_grids(
        save_dir, polygon, grid_width=2560, step_size=1280,
        grid_res=10, crs=rio.CRS.from_epsg(4326), **kwargs
):
    if not save_dir.exists():
        save_dir.mkdir()
    else:
        delete_directory_contents(save_dir)
    geod = Geod(ellps="WGS84")
    shapely.prepare(polygon)
    w, s, e, n = polygon.bounds
    mid_lat = (n + s)/2
    _, new_lat, _ = geod.fwd(lons=w, lats=mid_lat, dist=grid_res, radians=False, az=0)
    y_res = abs(mid_lat - new_lat)
    s -= y_res*grid_width/2
    n += y_res*grid_width/2

    mid_lat = (s + n)/2
    new_lon, _, _ = geod.fwd(lons=w, lats=mid_lat, dist=grid_res, radians=False, az=90)
    x_res = abs(w - new_lon)
    w -= x_res*grid_width/2
    e += x_res*grid_width/2
    if w < -179.999:
        w = -179.999
        e += x_res*grid_width/2
    elif e > 179.999:
        e = 179.999
        w -= x_res*grid_width/2

    height = grid_width*y_res
    total_rows = int(np.ceil((n - s)/y_res))
    num_steps = total_rows//step_size + 1
    for i in range(num_steps):
        min_lat = s + step_size*i*y_res
        max_lat = min_lat + height
        make_grids_for_fixed_lats(
            min_lat=min_lat, max_lat=max_lat, west=w, east=e, polygon=polygon,
            grid_width=grid_width, step_size=step_size, x_res=x_res, crs=crs,
            geod=geod, grid_res=grid_res, save_dir=save_dir
        )
        # break

if __name__ == '__main__':
    from water.basic_functions import get_country_bounding_box, name_to_box
    # make_grids('united_states', bbox=get_country_bounding_box('united_states'), grid_width=832,
    #            step_size=832, save_dir_name='grid', grid_res=10, crs=rio.CRS.from_epsg(4269))

    # grids_path = ppaths.country_data/'united_states/grid'
    # sentinel_path = ppaths.country_data/'united_states/sentinel'
    grids_path = ppaths.waterway/'sentinel'
    sentinel_path = ppaths.waterway/'waterways_burned'
    missing_files = []
    for file in grids_path.iterdir():
        if not (sentinel_path/file.name).exists():
            print(file)
            missing_files.append(name_to_box(file.name))
        #     break
        # if '34.6738342478' in file.name.split('_')[2]:
        #     missing_files.append(file)
    gpd.GeoSeries(missing_files, crs=4269).plot()
    # print(len(missing_files))
    # missing_files.sort(key=lambda x: x.name.split('_')[1])
    # with rio.open(missing_files[0]) as rio_f:
    #     old_bounds = rio_f.bounds
    # for file in missing_files[1:]:
    #     with rio.open(file) as rio_f:
    #         new_bounds = rio_f.bounds
    #         if old_bounds[0] != new_bounds[2]:
    #             print(file)
    #             print(old_bounds)
    #             print(new_bounds)
    #         old_bounds = new_bounds
            # print(tuple(rio_f.bounds))
    # data = np.array([[1,2,3], [4,5,6], [7,8,9]])
    # test = xr.DataArray(data=data, coords={'y':[0,.1,.2], 'x':[-.1,-.2,-.3]}, dims=['y', 'x'])
    # print(test)




# def make_overlap_grids()