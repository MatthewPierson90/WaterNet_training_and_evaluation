from water.basic_functions import multi_download, single_download, SharedMemoryPool
from water.paths import ppaths, Path
import pandas as pd
import geopandas as gpd


def extract_dl_links(text_file):
    with open(text_file, 'r') as f:
        text = f.read()
    dl_links = text.split('\n')
    dl_links = [link for link in dl_links if link != '']
    return dl_links


def nhd_to_parquet(hu4_index: int):
    file_path = ppaths.hu4_data/f'NHD_H_{hu4_index:04d}_HU4_GPKG.gpkg'
    parquet_path = ppaths.hu4_parquet / f'hu4_{hu4_index:04d}.parquet'
    if file_path.exists() and not parquet_path.exists():
        layers = ['NHDFlowline', 'NHDWaterbody', 'NHDArea', 'NHDLine']
        gdfs = []
        for layer in layers:
            gdf = gpd.read_file(file_path, layer=layer)
            gdf = gdf[['visibilityfilter', 'ftype', 'fcode', 'fcode_description', 'geometry']]
            gdfs.append(gdf)
        gdf = pd.concat(gdfs, ignore_index=True)
        gdf.to_parquet(parquet_path)


def download_nhd_hu4_data(num_proc: int=20):
    urls = extract_dl_links(Path(__file__).parent.joinpath('hu4_download_links.txt'))
    file_names = [ppaths.hu4_data/url.split('/')[-1] for url in urls]
    multi_download(
        urls, file_names, extract_zip_files=True, delete_zip_files_after_extraction=True, num_proc=num_proc
    )
    SharedMemoryPool(
        func=nhd_to_parquet, input_list=[i for i in range(100, 1900)], num_proc=num_proc
    ).run()


def download_wsb_data():
    url = extract_dl_links(Path(__file__).parent.joinpath('wsb_download_link.txt'))[0]
    download_save_path = ppaths.training_data/'WBD_National_GDB.zip'
    if not download_save_path.exists():
        single_download(url, download_save_path, extract_zip_files=False)
    file_path = ppaths.training_data/f'WBD_National_GDB.zip!WBD_National_GDB.gdb'
    gdf = gpd.read_file(file_path, layer='WBDHU4')
    gdf['hu4_index'] = gdf['huc4'].astype(int)
    gdf = gdf[gdf.hu4_index.apply(lambda x: (ppaths.hu4_parquet/f'hu4_{x}.parquet').exists())]
    gdf = gdf.sort_values(by='hu4_index').reset_index(drop=True)
    gdf['hu4_string'] = gdf['huc4']
    gdf[['hu4_index', 'hu4_string', 'geometry']].to_parquet(ppaths.hu4_hulls_parquet)
    for hu4_string in gdf.hu4_string:
        sub_gdf = gdf[gdf.hu4_string == hu4_string].reset_index(drop=True)
        sub_gdf.to_parquet(ppaths.hu4_hull/f'hu4_{hu4_string}.parquet')


if __name__ == '__main__':
    download_nhd_hu4_data()
    download_wsb_data()
