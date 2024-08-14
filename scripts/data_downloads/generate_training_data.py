import geopandas as gpd
import shapely
from water.paths import ppaths
# As a general note, you should only use at most the number of processes as your computer has
# processors/virtual processors. Even fewer if you want to use the machine while the process is running.
# Take note of the amount of memory being used, generally fewer processes require less memory.

num_proc = 30

hu4_hulls = gpd.read_parquet(ppaths.hu4_hulls_parquet)
polygon_list = hu4_hulls.geometry.tolist()
polygon = shapely.unary_union(polygon_list)


# Make Reference Grids
from water.data_functions.prepare.reference_grids import make_reference_grids
make_reference_grids(
    save_dir=ppaths.training_data/'reference_grids', polygon=polygon, num_proc=num_proc,
    grid_width=832, step_size=832, grid_res=10, crs=hu4_hulls.crs
)

# Cut Data to Reference Grids
from water.data_functions.prepare.cut_data import cut_data_to_match_file_list
from rasterio.warp import Resampling

# The sentinel files are hardly being reprojected, 4326 and 4269 over the us are similar,
# and the resolution isn't changing much, hence the nearest resampling
cut_data_to_match_file_list(
    save_dir=ppaths.sentinel_cut, data_dir=ppaths.sentinel_merged, resampling=Resampling.nearest,
    file_paths=ppaths.training_data/'reference_grids', num_proc=num_proc
)
# The elevation files are being upsampled from roughly 30m to 10m, and so we are using cubic sampling. I never
# got around to just trying nearest, and it might be worth it since the model may be able to do some of the upsampling,
# but cubic has worked out well.

cut_data_to_match_file_list(
    save_dir=ppaths.elevation_cut, data_dir=ppaths.elevation_data, resampling=Resampling.cubic,
    file_paths=ppaths.training_data/'reference_grids', num_proc=num_proc
)

# Burn Waterways
from water.data_functions.prepare.burn_waterway_rasters import do_multiple_inds

do_multiple_inds(range(101, 1900), num_proc=num_proc)

# Cut Training Data
from water.data_functions.prepare.cut_training_data import make_training_data_multi

parent_dirs = [ppaths.waterways_burned, ppaths.sentinel_cut, ppaths.elevation_cut]
make_training_data_multi(
    parent_dirs=parent_dirs, save_dir_path=ppaths.model_inputs_832, num_proc=num_proc,
    slice_width=832, row_step_size=832, col_step_size=832
)

make_training_data_multi(
    parent_dirs=parent_dirs, save_dir_path=ppaths.model_inputs_224, num_proc=num_proc,
    slice_width=224, row_step_size=224 - 22, col_step_size=224 - 22
)

# Make and move test and val data
from water.data_functions.prepare.make_test_val_lists import move_hu4_test_and_val_data

# These were our holdout indices, if left as None, then it will pick a new set of random indices
hu4_index_list = [103,204,309,403,505,601,701,805,904,1008,1110,1203,1302,1403,1505,1603,1708,1804]
# hu4_index_list = None
move_hu4_test_and_val_data(hu4_index_list)