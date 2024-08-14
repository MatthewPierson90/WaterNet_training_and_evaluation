import geopandas as gpd
import shapely
from water.paths import ppaths
from water.basic_functions import tt, time_elapsed
# As a general note, you should only use at most the number of processes as your computer has
# processors/virtual processors. Even fewer if you want to use the machine while the process is running.
# Take note of the amount of memory being used, generally fewer processes require less memory.

num_proc_download = 8
num_proc_merge = 8 #this process can use alot of memory, lower this if there are issues.

hu4_hulls = gpd.read_parquet(ppaths.hu4_hulls_parquet)
polygon_list = hu4_hulls.geometry.tolist()
polygon = shapely.unary_union(polygon_list)

# Download sentinel data for USA.
# It may be worthwhile to run this a few times, sometimes the connections timeout or get corrupted.
from water.data_functions.download.download_mpc_data import download_tiles_intersecting_polygon, CloudSearchInfo

mpc_input = dict(
    polygon=polygon, num_proc=num_proc_download, time_of_interest='2023-03-01/2023-10-31',
    save_dir_path=ppaths.sentinel_unmerged, cloud_info=CloudSearchInfo(1, 50, 5),
    max_items=4, min_year=2022, min_intersection_percent=.5,
    max_percent_remaining=.5, force_download=False
)

# It may be worthwhile to run this a few times, sometimes the connections timeout or get corrupted.
download_tiles_intersecting_polygon(**mpc_input)
# download_tiles_intersecting_polygon(**mpc_input)
# download_tiles_intersecting_polygon(**mpc_input)


# Merge Sentinel data for USA, if you don't want to keep the unmerged data, use merge_save_and_remove.
# That said, after waiting for all of that data to download, it might be worthwhile to merge and save, check how
# all the merged data looks, then remove the unmerged data.
from water.data_functions.download.merge_sentinel_tiles import merge_and_save_multi, merge_save_and_remove_multi

merge_inputs = dict(
    base_dir=ppaths.sentinel_unmerged, save_dir=ppaths.sentinel_merged, num_proc=num_proc_merge, num_steps=500
)
merge_and_save_multi(**merge_inputs)
# merge_save_and_remove_multi(**merge_inputs)

# Download elevation data
from water.data_functions.download.download_elevation import download_polygon_list_elevation_data

elevation_input = dict(
    polygon_list=polygon_list, num_proc=num_proc_download, save_dir=ppaths.elevation_data,
    print_progress=True, force_download=False
)
download_polygon_list_elevation_data(**elevation_input)
