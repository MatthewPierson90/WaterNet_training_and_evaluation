from water.paths import ppaths
from water.basic_functions import printdf
import geopandas as gpd


gdf = gpd.read_parquet(ppaths.evaluation_data/'basins_level_2_with_length/1020035180.parquet')
