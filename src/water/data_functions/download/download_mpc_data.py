from water.basic_functions import (SharedMemoryPool, geometrylike, tt, time_elapsed)
from water.paths import ppaths
import numpy as np
import planetary_computer as pc
import pystac_client
import stackstac
import xarray
import geopandas as gpd
import rasterio as rio
from rasterio import features
from pathlib import Path
import os
import shapely
import warnings
from dataclasses import dataclass

warnings.filterwarnings(action='ignore', category=UserWarning)

planetary_computer_stac = "https://planetarycomputer.microsoft.com/api/stac/v1"


def get_item_scl(item, bbox):
    scl = stackstac.stack(
            items=[pc.sign(item).to_dict()], assets=["SCL"],
            sortby_date=False, epsg=4326, bounds_latlon=bbox
    ).fillna(0).where(lambda x: x.isin([2, 4, 5, 6, 7, 11]), other=0).astype(np.uint8)[0].persist()
    scl = scl.where(scl == 0, other=1)
    return scl


@dataclass
class CloudSearchInfo:
    cloud_percent: int
    max_cloud_percent: int
    step_size: int
    
    def step(self):
        return CloudSearchInfo(
            cloud_percent=min(self.max_cloud_percent, self.cloud_percent + self.step_size),
            max_cloud_percent=self.max_cloud_percent, step_size=self.step_size
        )


class ItemFinder:
    def __init__(
            self, geometry: geometrylike,
            min_intersection_percent: float = .25,
            max_items: int = 10,
            min_year: int = 2023,
            catalog=None,
            max_percent_remaining=.05,
    ):
        if catalog is None:
            catalog = pystac_client.Client.open(planetary_computer_stac)
        self.catalog = catalog
        self.items = []
        self._seen = {}
        self.min_year = min_year
        self.max_items = max_items
        self.min_intersection_percent = min_intersection_percent
        self.geometry = geometry
        self.bounds = tuple(geometry.bounds)
        self.init_area = self.geometry.area
        self.array = None
        self.percent_found = 0
        self.total = None
        self.max_percent_remaining = max_percent_remaining

    def make_query(self, cloud_percent):
        return {'eo:cloud_cover': {'lt': cloud_percent}}
    
    def get_item_array(self, item):
        item_array = get_item_scl(item, self.geometry.bounds)
        return item_array
    
    @staticmethod
    def increase_toi(time_of_interest):
        year = int(time_of_interest[:4]) - 1
        time_split = time_of_interest.split('/')
        to_return = f'{year}{time_split[0][4:]}/{year}{time_split[1][4:]}'
        return to_return
    
    def make_search_kwargs(self, time_of_interest, cloud_percent):
        query = self.make_query(cloud_percent)
        search_kwargs = dict(
                collections=['sentinel-2-l2a'], datetime=time_of_interest, query=query
        )
        return search_kwargs
    
    def search_items(self, time_of_interest, cloud_percent):
        search_kwargs = self.make_search_kwargs(time_of_interest, cloud_percent)
        search = self.catalog.search(
                **search_kwargs
        )
        possible_items = list(search.get_items())
        possible_items.sort(
                key=lambda x: x.properties['eo:cloud_cover'] + x.properties['s2:nodata_pixel_percentage']
        )
        return possible_items


    def make_init_array_total_and_percent_found(self, item_array):
        self.array = features.rasterize(
            shapes=((self.geometry, 2)), out_shape=item_array.shape[1:],
            transform=item_array.rio.transform(), all_touched=False, dtype=int
        )
        self.array[self.array > 0] = -1
        self.array += 1
        self.array = np.stack([self.array])
        self.total = self.array.shape[-1]*self.array.shape[-2]
        self.percent_found = self.array.sum()/self.total


    def check_possible_items(self, possible_items):
        for item in possible_items:
            gran_id = item.properties['s2:granule_id']
            if not gran_id in self._seen:
                self._seen[gran_id] = True
                item_polygon = shapely.box(*item.bbox)
                if item_polygon.intersects(self.geometry):
                    item_array = get_item_scl(item, self.geometry.bounds)
                    if self.array is None:
                        self.make_init_array_total_and_percent_found(item_array)
                    item_array = item_array.to_numpy().astype(int)
                    new_array = np.concatenate([item_array, self.array], axis=0).max(axis=0, keepdims=True)
                    new_percent = new_array.sum()/self.total
                    print(new_percent - self.percent_found, self.min_intersection_percent*(1-self.percent_found))
                    if (new_percent - self.percent_found) >= self.min_intersection_percent*(1-self.percent_found):
                        self.array = new_array
                        self.percent_found = new_percent
                        self.items.append(item)
                    if 1 - self.percent_found < 1e-4 or len(self.items) >= self.max_items:
                        return self.items
        return self.items

    def find_items(
            self, time_of_interest: str = f'2023-01-01/2023-06-30',
            cloud_info: CloudSearchInfo = CloudSearchInfo(1, 10, 5), remove_clouds=True, is_init=True
    ):
        possible_items = self.search_items(time_of_interest, cloud_info.cloud_percent)
        self.check_possible_items(possible_items=possible_items)
        if not (1 - self.percent_found < self.max_percent_remaining/5 or len(self.items) >= self.max_items):
            if cloud_info.cloud_percent < cloud_info.max_cloud_percent:
                cloud_info = cloud_info.step()
                self.items = self.find_items(
                    time_of_interest=time_of_interest, cloud_info=cloud_info, remove_clouds=remove_clouds, is_init=False
                )
            else:
                year = int(time_of_interest[:4])
                if year > self.min_year:
                    time_of_interest = self.increase_toi(time_of_interest)
                    self.items = self.find_items(
                        time_of_interest=time_of_interest, cloud_info=cloud_info, remove_clouds=remove_clouds, is_init=False
                    )
        if 1-self.percent_found < 1 and is_init:
            print(self.bounds, 1 - self.percent_found, self.max_percent_remaining, len(self.items))
        if 1 - self.percent_found < self.max_percent_remaining:
            return self.items
        else:
            return []


class TileFinder(ItemFinder):
    def __init__(self, tile: str,
                 geometry: geometrylike,
                 min_intersection_percent: float = .25,
                 max_items: int = 10,
                 min_year: int = 2020,
                 max_percent_remaining: float = .05,
                 catalog=None):
        super().__init__(
                geometry=geometry, max_items=max_items, min_year=min_year, catalog=catalog,
                min_intersection_percent=min_intersection_percent, max_percent_remaining=max_percent_remaining
        )
        self.tile = tile
    
    def make_query(self, cloud_percent):
        return {'eo:cloud_cover': {'lt': cloud_percent}, 's2:mgrs_tile': {'eq': self.tile}}


def save(data: 'xarray.DataArray',
         save_path: Path,
         crs,
         transform,
         ) -> None:
    kwargs = {
        'driver': 'GTiff', 'dtype': 'uint8', 'nodata': 0, 'crs': crs, 'transform': transform,
        'width': data.shape[2], 'height': data.shape[1], 'count': data.shape[0]
    }
    with rio.open(save_path, 'w', **kwargs) as dst:
        dst.write(data.to_numpy())


def clip_rbg(rgb):
    rgb_clip = (255/(1 + np.exp(-.6*rgb)))
    return rgb_clip


def shift_color(data, means, stds):
    for day in range(data.shape[0]):
        for ch in range(4):
            if stds[day, ch] == 0:
                stds[day, ch] = 1
    data = (data - means)/stds
    data = clip_rbg(data)
    return data


def shift_color_full(sen_data):
    scl = sen_data[:, 4]
    sen_data = sen_data[:, 0:4, :, :].astype(np.float32)
    sen_data = sen_data.where(sen_data > 0, other=np.nan)
    means = sen_data.where(scl.isin([2, 4, 5, 6, 7, 11]), other=np.nan).mean(dim=('y', 'x'), skipna=True)
    stds = sen_data.where(scl.isin([2, 4, 5, 6, 7, 11]), other=np.nan).std(dim=('y', 'x'), skipna=True)
    sen_data = shift_color(sen_data, means, stds)
    return sen_data


def download_sentinel_by_tile(
        tile: str, tile_geometry: shapely.Polygon, time_of_interest: str = '2016-01-01/2023-12-30',
        save_dir_path: Path = ppaths.sentinel_unmerged,
        catalog=None, cloud_info=CloudSearchInfo(1, 50, 10), max_items=6, min_year=2020,
        min_intersection_percent=.1, max_percent_remaining=.05, force_download=False,
):
    gw, gs, ge, gn = tile_geometry.bounds
    save_subdir = save_dir_path/f'bbox_{gw}_{gs}_{ge}_{gn}'
    merged_path = ppaths.sentinel_merged/f'bbox_{gw}_{gs}_{ge}_{gn}.tif'
    if not save_subdir.exists():
        save_subdir.mkdir()
    tile_finder = TileFinder(
        tile=tile, geometry=tile_geometry, catalog=catalog, max_items=max_items, min_year=min_year,
        min_intersection_percent=min_intersection_percent, max_percent_remaining=max_percent_remaining
    )
    items = tile_finder.find_items(time_of_interest=time_of_interest, cloud_info=cloud_info)
    if not merged_path.exists() or force_download:
        if len(items) > 0:
            signed_items = [
                pc.sign(item).to_dict() for item in items if not (save_subdir/f'{item.id}.tif').exists()
            ]
            print(f'num items: {len(items)}, num new items: {len(signed_items)}')
            if len(signed_items) > 0:
                data = stackstac.stack(
                        signed_items, epsg=4326, dtype=np.uint16, fill_value=0, bounds_latlon=tile_geometry.bounds,
                        assets=["B08", "B04", "B03", "B02", 'SCL'], sortby_date=False,
                ).persist()
                crs = data.crs
                transform = data.transform
                scl = data[:, -1:].copy().astype(np.uint8)
                shifted = shift_color_full(data).fillna(0).astype(np.uint8)
                for ind, item in enumerate(signed_items):
                    id = item['id']
                    item_data = shifted[ind].copy()
                    scl_data = scl[ind].copy()
                    save_as = save_subdir/f'{id}.tif'
                    scl_save_as = save_subdir/f'{id}_scl.tif'
                    save(data=item_data, save_path=save_as, crs=crs, transform=transform)
                    save(data=scl_data, save_path=scl_save_as, crs=crs, transform=transform)


def get_tiles_for_polygon(polygon):
    world_tiles = gpd.read_parquet(ppaths.sentinel_tiles_parquet)
    str_tree = shapely.STRtree(world_tiles.geometry)
    to_return = world_tiles.loc[str_tree.query(polygon.buffer(.15), predicate='intersects')]
    to_return['geometry'] = to_return.buffer(-2e-7)
    to_return = zip(to_return.Name, to_return.geometry)
    return to_return



def get_tiles_for_continent(continent: str):
    """
    Get all tiles for a given continent. Continents can be from the following list:
    'Oceania', 'Asia', 'Europe', 'Americas', 'Africa', 'Antarctica'
    """
    world_boundaries = gpd.read_parquet(ppaths.world_boundaries_parquet)
    world_boundaries = world_boundaries[world_boundaries.continent == continent.lower().capitalize()]
    tile_list = []
    for geometry in world_boundaries.geometry:
        tile_list += get_tiles_for_polygon(geometry)
    return tile_list



def download_tile_list(
        tile_name_geometry_list, num_proc, time_of_interest=f'2023-01-01/2023-06-30',
        save_dir_path=ppaths.deploy_data/'sentinel_4326', **kwargs
):
    inputs = []
    tiles_seen = set()
    save_subdirs_seen = set()
    print(f'num seen: {len(save_subdirs_seen)}')
    for tile_name, tile_geometry in tile_name_geometry_list:
        gw, gs, ge, gn = tile_geometry.bounds
        save_subdir = f'bbox_{gw}_{gs}_{ge}_{gn}'
        save_path = save_dir_path / save_subdir
        if not save_subdir in save_subdirs_seen:
            if not save_path.exists():
                save_path.mkdir()
            # if tile_name not in tiles_seen:
            if len(list(save_path.iterdir())) == 0:
                print(save_subdir)
                save_subdirs_seen.add(save_subdir)
                next_inputs = {
                    'tile': tile_name, 'tile_geometry': tile_geometry,
                    'time_of_interest': time_of_interest, 'save_dir_path': save_dir_path,
                }
                next_inputs.update(**kwargs)
                inputs.append(next_inputs)
    print(len(inputs))
    np.random.shuffle(inputs)
    if num_proc > 1:
        SharedMemoryPool(
            func=download_sentinel_by_tile, input_list=inputs, time_out=300,
            num_proc=num_proc, use_kwargs=True, terminate_on_error=False
        ).run()
    else:
        for input in inputs:
            download_sentinel_by_tile(**input)


def download_tiles_intersecting_polygon(
        polygon: shapely.Polygon, num_proc: int, time_of_interest=f'2023-01-01/2023-06-30',
        save_dir_path=ppaths.sentinel_unmerged, **kwargs
):
    tiles_list = get_tiles_for_polygon(polygon)
    download_tile_list(tiles_list, num_proc, time_of_interest, save_dir_path, **kwargs)


def download_continent_tiles(continent, num_proc,
                             time_of_interest=f'2023-01-01/2023-06-30',
                             save_dir_path=ppaths.sentinel_unmerged,
                             **kwargs
                             ):
    if not save_dir_path.exists():
        save_dir_path.mkdir(parents=True)
    tiles_geometry = get_tiles_for_continent(continent)
    download_tile_list(tiles_geometry, num_proc, time_of_interest, save_dir_path, **kwargs)






if __name__ == '__main__':
    from water.basic_functions import wait_n_seconds
    # inputs = dict(
    #     num_proc=8, time_of_interest='2023-03-01/2023-10-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_africa',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=6, min_year=2020, min_intersection_percent=.4,
    #     max_percent_remaining=.9
    # )
    # download_continent_tiles('africa', **inputs)
    #
    # inputs = dict(
    #     continent='Africa',
    #     num_proc=8, time_of_interest='2023-01-01/2023-12-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_africa',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.01
    # )
    # download_continent_tiles(**inputs)
    #
    # inputs = dict(
    #     continent='Africa',
    #     num_proc=8, time_of_interest='2023-01-01/2023-12-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_africa',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.03
    # )
    # download_continent_tiles(**inputs)
    #
    # inputs = dict(
    #     continent='Africa',
    #     num_proc=8, time_of_interest='2023-01-01/2023-12-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_africa',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.05
    # )
    # download_continent_tiles(**inputs)
    #
    # inputs = dict(
    #     continent='Africa',
    #     num_proc=8, time_of_interest='2023-01-01/2023-12-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_africa',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.1
    # )
    # download_continent_tiles(**inputs)
    #
    # inputs = dict(
    #     continent='Africa',
    #     num_proc=8, time_of_interest='2023-01-01/2023-12-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_africa',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.2
    # )
    # download_continent_tiles(**inputs)
    #
    # inputs = dict(
    #     continent='Africa',
    #     num_proc=8, time_of_interest='2023-01-01/2023-12-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_africa',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.5
    # )
    # download_continent_tiles(**inputs)




    # inputs = dict(
    #     continent='Europe',
    #     num_proc=8, time_of_interest='2023-03-01/2023-10-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_europe',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.01
    # )
    # download_continent_tiles(**inputs)

    # inputs = dict(
    #     continent='Europe',
    #     num_proc=8, time_of_interest='2023-03-01/2023-10-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_europe',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.03
    # )
    # download_continent_tiles(**inputs)
    #
    #
    # inputs = dict(
    #     continent='Europe',
    #     num_proc=8, time_of_interest='2023-03-01/2023-10-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_europe',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.05
    # )
    # download_continent_tiles(**inputs)
    #
    #
    # inputs = dict(
    #     continent='Europe',
    #     num_proc=8, time_of_interest='2023-03-01/2023-10-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_europe',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.1
    # )
    # download_continent_tiles(**inputs)
    #
    #
    # inputs = dict(
    #     continent='Europe',
    #     num_proc=8, time_of_interest='2023-03-01/2023-10-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_europe',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.2
    # )
    # download_continent_tiles(**inputs)
    #
    # inputs = dict(
    #     continent='Europe',
    #     num_proc=8, time_of_interest='2023-03-01/2023-10-30',
    #     save_dir_path=ppaths.deploy_data / 'sentinel_tiles_europe',
    #     cloud_info=CloudSearchInfo(1, 50, 10),
    #     max_items=4, min_year=2022, min_intersection_percent=.5,
    #     max_percent_remaining=.5
    # )
    # download_continent_tiles(**inputs)


    # country_list = [
    #     'syria', 'iraq', 'iran', 'lebanon', 'jordan', 'israel', 'saudi_arabia', 'oman', 'yemen', 'kuwait',
    #     'bahrain', 'qatar', 'united_arab_emirates', 'armenia', 'georgia', 'azerbaijan', 'turkmenistan',
    #     'uzbekistan', 'afghanistan', 'pakistan', 'tajikistan', 'kyrgyzstan', 'kazakhstan'
    # ]
    # country_list = [
    #     'jammu-kashmir', 'aksai chin', 'arunachal pradesh'
    # ]
    # tiles = list(get_tiles_for_polygon(polygon))
    # storage_dir = ppaths.training_data/'waterway_storage/sentinel_tiles_america'
    # tiles_keep = []
    # save_dir = ppaths.deploy_data/'sentinel_tiles_america'
    # for tile_name, tile_geometry in tiles:
    #     gw, gs, ge, gn = tile_geometry.bounds
    #     save_subdir = f'bbox_{gw}_{gs}_{ge}_{gn}'
    #     if not (storage_dir/save_subdir).exists():
    #         tiles_keep.append((tile_name, tile_geometry))
    #     else:
    #         os.system(f'cp -r {storage_dir/save_subdir} {save_dir}')
    # print(len(tiles_keep))
    #
    # download_tile_list(
    #     tile_name_geometry_list=tiles_keep,
    #     num_proc=8,
    #     time_of_interest='2023-03-01/2023-9-30',
    #     save_dir_path=ppaths.deploy_data/'sentinel_tiles_america'
    # )

    world_data = gpd.read_parquet(ppaths.world_info/'world_boundaries.parquet')
    polygons = []
    for geom in world_data.geometry:
        if hasattr(geom, 'geoms'):
            polygons.extend(geom.geoms)
        else:
            polygons.append(geom)
    print(len(polygons))
    s = tt()
    polygon = shapely.unary_union(polygons)
    tiles = list(get_tiles_for_polygon(polygon))
    print(len(tiles))
    time_elapsed(s, 2)


    download_inputs = dict(
        num_proc=8, time_of_interest='2023-01-01/2023-12-30',
        save_dir_path=ppaths.deploy_data / 'sentinel_tiles_africa',
        cloud_info=CloudSearchInfo(1, 90, 10),
        max_items=4, min_year=2023, min_intersection_percent=.03,
        max_percent_remaining=.99
    )
    download_tile_list(tiles, **download_inputs)


    download_inputs = dict(
        num_proc=8, time_of_interest='2023-01-01/2023-12-30',
        save_dir_path=ppaths.deploy_data / 'sentinel_tiles_africa',
        cloud_info=CloudSearchInfo(1, 90, 10),
        max_items=4, min_year=2022, min_intersection_percent=.03,
        max_percent_remaining=.99
    )
    download_tile_list(tiles, **download_inputs)
