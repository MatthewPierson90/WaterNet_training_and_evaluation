from water.basic_functions import single_download
from water.paths import ppaths
import geopandas as gpd
import shapely
### Download Sentinel Tiles

url_sen = 'https://hls.gsfc.nasa.gov/wp-content/uploads/2016/03/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml'
save_as_sen = ppaths.world_info/'sentinel_tiles.kml'
if not save_as_sen.exists():
    single_download(url_sen, save_as_sen)
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
gdf_sen = gpd.read_file(save_as_sen, driver='KML')
gdf_sen['geometry'] = shapely.force_2d(gdf_sen.geometry).apply(lambda x: shapely.unary_union(x.geoms))
print(gdf_sen.crs)
gdf_sen.to_parquet(ppaths.sentinel_tiles_parquet)

### Download World Boundaries

url_wb = 'https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip'
save_as_wb = ppaths.world_info/'ne_110m_admin_0_countries.zip'
if not save_as_wb.exists():
    single_download(url_wb, save_as_wb)
gdf_wb = gpd.read_file(save_as_wb)
print(gdf_wb.crs)
gdf_wb.to_parquet(ppaths.world_boundaries_parquet)