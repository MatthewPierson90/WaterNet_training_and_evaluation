import rasterio as rio
from rasterio import features
import geopandas as gpd
import shapely
import numpy as np
from water.basic_functions import SharedMemoryPool
from water.paths import ppaths


def open_hu4_data(index:int,
                  only_linestrings=False):
    waterway_path = ppaths.training_data/f'hu4_parquet/hu4_{index:04d}.parquet'
    if waterway_path.exists():
        gdf = gpd.read_parquet(waterway_path)
        if only_linestrings:
            gdf = gdf[gdf.type.isin(['LineString', 'MultiLineString'])].reset_index(drop=True)
            gdf = gdf[
                gdf.fcode_description.str.contains('Stream') | gdf.fcode_description.str.contains('Artificial')
            ].reset_index(drop=True)
        if len(gdf)>0:
            return gdf


def get_waterways_in_bbox(bbox,
                          waterways):
    waterways_str = shapely.STRtree(waterways.geometry)
    box = shapely.box(*bbox)
    intersects_box_list = waterways_str.query(box, 'intersects')
    water_in_box = waterways.loc[intersects_box_list].reset_index(drop=True)
    water_in_box['geometry'] = shapely.intersection(box, water_in_box.geometry)
    return water_in_box


def find_intersection_points(water_gdf: gpd.GeoDataFrame):
    line_strings = water_gdf[water_gdf.geom_type == 'LineString'].geometry.to_list()
    multi_line_strings = water_gdf[water_gdf.geom_type == 'MultiLineString'].geometry.to_list()
    for multi in multi_line_strings:
        line_strings.extend(multi.geoms)
    if len(line_strings) == 0:
        return []
    head_points = shapely.line_interpolate_point(line_strings, 0, normalized=True)
    tail_points = shapely.line_interpolate_point(line_strings, 1, normalized=True)
    all_points = np.concatenate([head_points, tail_points])
    points_tree = shapely.STRtree(all_points)
    line_inds, points_inds = points_tree.query(line_strings, predicate='intersects')
    unique_inds, inds_counts = np.unique(points_inds, return_counts=True)
    return all_points[unique_inds[np.where(inds_counts > 2)]]


def burn_to_raster_loop(water_gdf: gpd.GeoDataFrame, out_shape, transform, init_buffer=.0005):
    to_burn = []
    if len(water_gdf) > 0:
        for i in range(1, water_gdf.water_type.max() + 1):
            geometry = shapely.unary_union(water_gdf[water_gdf.water_type == i].geometry.to_numpy())
            to_burn.append((geometry.buffer(init_buffer), i))
        image = features.rasterize(
            to_burn, all_touched=True, out_shape=out_shape, transform=transform
        )
        intersection_points = find_intersection_points(water_gdf)
        if len(intersection_points) > 0:
            intersection_geometry = shapely.unary_union(shapely.buffer(intersection_points, init_buffer*4))
            intersection_raster = features.rasterize(
                (intersection_geometry, 1), all_touched=True, out_shape=out_shape, transform=transform
            )
            image[(image > 0) & (intersection_raster == 1)] = 22
    else:
        image = np.zeros(out_shape, dtype=np.uint8)
    return image


def burn_waterway_raster(raster_path,
                         water_in_box: gpd.GeoDataFrame,
                         dst_profile,
                         out_shape,
                         transform,
                         save_dir_name='waterways_burned',
                         ):
    image = burn_to_raster_loop(
        water_gdf=water_in_box, out_shape=out_shape, init_buffer=0.00005, transform=transform,
    )
    image = np.stack([image], axis=0, dtype=np.uint8)
    dst_profile['count'] = 1
    dst_profile['dtype'] = np.uint8
    dst_profile['nodata'] = None
    burned_dir = ppaths.training_data/save_dir_name
    if not burned_dir.exists():
        burned_dir.mkdir()
    save_dir = burned_dir
    if not save_dir.exists():
        save_dir.mkdir()
    save_path = save_dir/raster_path.name
    if save_path.exists():
        with rio.open(save_path) as src_f:
            data = src_f.read()
            data = np.stack([data, image], axis=0)
            data = data.max(axis=0)
            image = data
    with rio.open(save_path, 'w', **dst_profile) as dst_f:
        dst_f.write(image)
    return image


def set_water_type(fcode_description: str,
                   types: list[tuple[str, int]] = (('Ephemeral', 1), ('Intermittent', 2))
                   ) -> int:
    max_val = 0
    fcode_description = fcode_description.lower()
    for type, val in types:
        type = type.lower()
        max_val = val if val > max_val else max_val
        if type in fcode_description:
            return val
    return max_val + 1


ignored_types = (
    'Ice Mass', 'Bridge', 'Tunnel', 'Underground Conduit', 'Submerged Stream', 'Levee', 'Nonearthen Shore', 'Rapids',
    'Dam/Weir', 'Gate', 'Pipeline'
)


def get_hu4_waterways(hu4_index: int,
                      types_to_remove: iter = ignored_types,
                      types_to_differentiate: list[tuple[str, int]] = (
                              ('Playa', 1),
                              ('Inundation', 2),
                              ('Swamp/Marsh: Hydrographic Category = Intermittent', 3),
                              ('Swamp/Marsh: Hydrographic Category = Perennial', 4),
                              ('Swamp', 5),
                              ('Reservoir', 6),
                              ('Lake/Pond: Hydrographic Category = Intermittent', 7),
                              ('Lake/Pond: Hydrographic Category = Perennial', 8),
                              ('Lake/Pond', 9),
                              ('Spillway', 10),
                              ('Drainageway', 11),
                              ('Wash', 12),
                              ('Canal Ditch: Canal Ditch Type = Stormwater', 13),
                              ('Canal/Ditch: Canal/Ditch Type = Aqueduct', 14),
                              ('Canal', 15),
                              ('Artificial Path', 16),
                              ('Stream/River: Hydrographic Category = Ephemeral', 17),
                              ('Stream/River: Hydrographic Category = Intermittent', 18),
                              ('Stream/River: Hydrographic Category = Perennial', 19),
                              ('Stream/River', 20),
                      )):
    wws = open_hu4_data(hu4_index)
    # print(wws)
    for type in types_to_remove:
        wws = wws[~wws.fcode_description.str.contains(type)].reset_index(drop=True)
    wws['water_type'] = wws.fcode_description.apply(
            lambda x: set_water_type(x, types_to_differentiate)
    )
    return wws


def make_bbox(file_name):
    if '.tif' in file_name:
        file_name = file_name.split('.tif')[0]
    bbox = tuple([float(piece) for piece in file_name.split('_')[1:]])
    return bbox


def do_files(files: list,
             water: gpd.GeoDataFrame,
             ww_shape: shapely.Polygon,
             save_dir_name='waterways_burned'):
    count = 0
    for file in files:
        box = make_bbox(file.name)
        if ww_shape.intersects(shapely.box(*box)):
            count += 1
            with rio.open(file) as src_f:
                dst_profile = src_f.profile
                out_shape = src_f.shape
                transform = src_f.transform
                e, s, w, n = src_f.bounds
            water_in_box = get_waterways_in_bbox((e, s, w, n), water)
            burn_waterway_raster(
                    file, water_in_box=water_in_box, save_dir_name=save_dir_name,
                    dst_profile=dst_profile, out_shape=out_shape, transform=transform
            )


def do_all_for_ind(ww_ind, num_proc, files):
    hu4_file = ppaths.training_data/f'hu4_parquet/hu4_{ww_ind:04d}.parquet'
    hull_file = ppaths.training_data/f'hu4_hull/hu4_{ww_ind:04d}.parquet'
    if hu4_file.exists() and hull_file.exists():
        print(f'Working on {ww_ind}')
        water = get_hu4_waterways(ww_ind)
        crs = water.crs
        np.random.shuffle(files)
        num_files = len(files)
        # num_steps = num_proc*5
        num_steps = num_proc
        step_size = num_files//(num_steps)

        ww_shape = gpd.read_parquet(hull_file).geometry[0]
        input_list = [
            {'files': files[i*step_size:(i + 1)*step_size],
             'water': water, 'ww_shape': ww_shape,
             } for i in range(num_steps)
        ]
        SharedMemoryPool(
            num_proc=num_proc, func=do_files, input_list=input_list, use_kwargs=True, sleep_time=0
        ).run()


def do_multiple_inds(inds_to_do, num_proc, init_dir='sentinel', exclude_test_val=True):
    if not (ppaths.training_data/'waterways_burned').exists():
        (ppaths.training_data/'waterways_burned').mkdir()
    # files = list((ppaths.training_data / init_dir).glob('*'))
    files = [
        file for file in (ppaths.training_data/init_dir).iterdir()
        if not (ppaths.training_data/f'waterways_burned/{file.name}').exists()
    ]
    print(len(files))
    # print(len(files))
    # if exclude_test_val:
    #     test_names = get_test_file_names()
    #     val_names = get_val_file_names()
    #     used_names = test_names + val_names
    #     used_names = set([f'{name}.tif' for name in used_names])
    #     files = [file for file in files if file.name in used_names]
    # files.sort(key=lambda x: x.name)
    # for file in files:
    #     print(file)
    for ind in inds_to_do:
        do_all_for_ind(ind, num_proc, files=files)

if __name__ == '__main__':
    from pprint import pprint
    # water_types = set()
    # for ww_ind in range(101, 1900):
        # hu4_file = ppaths.training_data / f'hu4_parquet/hu4_{ww_ind:04d}.parquet'
    #     hull_file = ppaths.training_data / f'hu4_hull/hu4_{ww_ind:04d}.parquet'
    #     if hu4_file.exists() and hull_file.exists():
    #         print(f'opening ww ind {ww_ind}')
    #         water = get_hu4_waterways(ww_ind)
    #         water_types.update(water.fcode_description.unique())
    # water_types = list(water_types)
    # water_types.sort()
    # for item in water_types:
    #     print(item)
    # pprint(water_types)
    do_multiple_inds(range(101, 1900), 25)

