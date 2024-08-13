from water.basic_functions import ppaths, printdf
import geopandas as gpd
import pandas as pd

total_gdf = None
hulls_to_keep = []
for file in (ppaths.training_data/'wbd_data').iterdir():
    hu4_file = file/'Shape/WBDHU4.shp'
    gdf = gpd.read_file(hu4_file)
    gdf = gdf[['huc4', 'geometry']]
    if total_gdf is None:
        total_gdf = gdf
    else:
        total_gdf = pd.concat([total_gdf, gdf], ignore_index=True)
    gdf = gdf.set_index('huc4')
    for hu4_index in gdf.index:
        if (ppaths.training_data/f'hu4_hull/hu4_{hu4_index}.parquet').exists():
            hulls_to_keep.append(int(hu4_index))
            gdf1 = gdf.loc[hu4_index:hu4_index].reset_index(drop=True)
            gdf1.to_parquet(ppaths.training_data/f'hu4_hull/hu4_{hu4_index}.parquet')

total_gdf = total_gdf.rename(columns={'huc4': 'hu4_index'})
total_gdf['hu4_index'] = total_gdf.hu4_index.astype(int)
total_gdf = total_gdf[total_gdf.hu4_index.isin(hulls_to_keep)]
total_gdf = total_gdf.sort_values(by='hu4_index')
total_gdf.to_parquet(ppaths.training_data/'hu4_hulls.parquet')
# print(gdf.crs)
# print(gdf.columns)
# printdf(gdf)