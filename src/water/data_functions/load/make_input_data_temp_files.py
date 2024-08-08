from water.basic_functions import ppaths
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import rasterio as rio
from rasterio import transform
import shapely
from pyproj import Geod
import numpy as np
from water.basic_functions import ppaths, get_country_polygon, delete_directory_contents


def make_grid(grid_width, min_lon, max_lon, min_lat, max_lat, country, save_dir_name):
    save_dir = ppaths.waterway/f'country_data/{country}/{save_dir_name}'
    file_name = f'bbox_{min_lon:.10f}_{min_lat:.10f}_{max_lon:.10f}_{max_lat:.10f}.tif'
    save_path = save_dir/file_name
    trans = transform.from_bounds(min_lon, min_lat, max_lon, max_lat, grid_width, grid_width)
    data = np.ones((1, grid_width, grid_width))
    meta = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': 0,
        'width': grid_width,
        'height': grid_width,
        'count': 1,
        'crs': rio.CRS.from_epsg(4326),
        'transform': trans
    }
    with rio.open(save_path, 'w', **meta) as dst_f:
        dst_f.write(data)


def make_grids_for_fixed_lats(min_lat, max_lat, west, east, x_res, grid_width,
                              step_size, country, save_dir_name, polygon):
    # mid_lat = (max_lat + min_lat)/2
    # new_lon, _, _ = geod.fwd(lons=west, lats=mid_lat, dist=grid_res, radians=False, az=90)
    # x_res = abs(west - new_lon)
    # west -= x_res*grid_width/2
    # east += x_res*grid_width/2
    width = grid_width*x_res
    total_cols = int(np.ceil((east - west)/x_res))
    num_steps = total_cols//step_size
    for i in range(num_steps):
        min_lon = west + step_size*i*x_res
        max_lon = min_lon + width
        if polygon.intersects(shapely.box(min_lon, min_lat, max_lon, max_lat)):
            make_grid(
                min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon,
                grid_width=grid_width, country=country, save_dir_name=save_dir_name
            )



def make_grids(country, bbox, grid_width=2560, step_size=1280, save_dir_name='temp', grid_res=10):
    country_dir = ppaths.waterway/f'country_data/{country}'
    save_dir = country_dir/f'{save_dir_name}'
    if not save_dir.exists():
        save_dir.mkdir()
    else:
        delete_directory_contents(save_dir)
    geod = Geod(ellps="WGS84")
    w, s, e, n = bbox
    country_polygon = get_country_polygon(country)
    shapely.prepare(country_polygon)
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

    height = grid_width*y_res
    total_rows = int(np.ceil((n - s)/y_res))
    num_steps = total_rows//step_size
    for i in range(num_steps):
        min_lat = s + step_size*i*y_res
        max_lat = min_lat + height
        make_grids_for_fixed_lats(
            min_lat=min_lat, max_lat=max_lat, west=w, east=e, country=country, polygon=country_polygon,
            grid_width=grid_width, step_size=step_size, save_dir_name=save_dir_name, x_res=x_res
        )