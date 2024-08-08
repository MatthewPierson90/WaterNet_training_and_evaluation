import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from water.basic_functions import ppaths, Path
from water.make_country_waterways.cut_data import name_to_box
from water.make_country_waterways.merge_prepared_data import make_slope
import shapely


def find_file_intersecting_point(
        point: shapely.Point,
        files: iter
)-> Path:
    for file in files:
        name = file.name
        file_bbox = name_to_box(name)
        if point.intersects(file_bbox):
            print(file)
            return file
    return Path('/334242/243823423/342234')


def plot_inputs_outputs_at_point(
        lon: float, lat: float, country:str
)-> plt.Axes:
    country_dir = ppaths.waterway/f'country_data/{country}'
    output_dir = country_dir/'output_data'
    sentinel_dir = country_dir/'sentinel_4326_cut'
    elevation_dir = country_dir/'elevation_cut'
    point = shapely.Point((lon, lat))
    output_file = find_file_intersecting_point(point,output_dir.glob('*'))
    sentinel_file = sentinel_dir/output_file.name
    elevation_file = elevation_dir/output_file.name
    if output_file.exists() and sentinel_file.exists() and elevation_file.exists():
        sentinel_file = sentinel_dir/output_file.name
        elevation_file = elevation_dir/output_file.name
        with rio.open(output_file) as out_f:
            out_data = out_f.read()[0]
            w,s,e,n = out_f.bounds
        with rio.open(sentinel_file) as sen_f:
            sen_data = sen_f.read()
            sen_data = sen_data[1:].transpose((1,2,0))
        with rio.open(elevation_file) as el_f:
            el_data = el_f.read()
            x_der, y_der = make_slope(el_data)
            xy_grad = np.sqrt(x_der**2 + y_der**2)
            print(xy_grad.shape)
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        ax[0].imshow(sen_data, extent=(w, e, s, n))
        ax[1].imshow(xy_grad[0], extent=(w, e, s, n), cmap='inferno', vmax=xy_grad.mean() + 2*xy_grad.std())
        ax[2].imshow(np.round(out_data), extent=(w, e, s, n), vmin=0, vmax=1)
    else:
        print(output_file.exists(), sentinel_file.exists(), elevation_file.exists())


if __name__ == '__main__':
    ll = (36.10, 7.875)
    ctry = 'ethiopia'
    plot_inputs_outputs_at_point(country=ctry, lon=ll[0], lat=ll[1])
    # out_cut = list((ppaths.waterway/'country_data/uganda/output_data_cut/').glob('*'))
    # boxes = []
    # for file in out_cut:
    #     boxes.append(name_to_box(file.name))
    # gpd.GeoSeries(boxes, crs=4326).plot()
    # file = out_cut[10]
    # file_name = file.name.split('.tif')[0]+'.parquet'
    # import geopandas as gpd
    # with rio.open(file) as rio_f:
    #     data = rio_f.read()
    #     w, s, e, n = rio_f.bounds
    #     x_res, y_res = rio_f.res
    # print(w,s,e,n, x_res, y_res)
    # fig, ax = plt.subplots()
    # ax.imshow(data[0], extent=(w,e,s,n))
    # gdf = gpd.read_parquet(ppaths.waterway/f'country_data/uganda/vector_data/{file_name}')
    # gdf.plot(ax=ax)
    # print(gdf)
