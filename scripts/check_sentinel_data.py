import rasterio as rio
import rioxarray as rxr
from water.basic_functions import ppaths, Path, tt, time_elapsed
import pandas as pd

def check_file(file_path: Path, df_data: dict):
    df_data['file_name'].append(file_path.name)
    with rio.open(file_path) as src:
        data = src.read()
        zero_count = len(data[data==0])
        total_count = data.shape[0]*data.shape[1]*data.shape[2]
        df_data['zero_count'].append(zero_count)
        df_data['total_count'].append(total_count)
        df_data['percent'].append(zero_count/total_count)
    return data


def check_dir_files(dir_path: Path):
    file_paths = list(dir_path.glob('*.tif'))
    num_files = len(file_paths)
    data = dict(file_name=[], zero_count=[], total_count=[], percent=[])
    s = tt()
    for ind, file_path in enumerate(file_paths):
        try:
            check_file(file_path, data)
        except:
            pass
        if ind % (num_files//100) == 0:
            print(f'Completed {(ind+1)/num_files:.0%} ({ind+1}/{num_files})')
            time_elapsed(s, 2)
    df = pd.DataFrame(data=data)
    df.to_csv(ppaths.country_data/'sentinel_4326_quality.csv', index=False)
    return df



if __name__ == '__main__':
    # df = check_dir_files(ppaths.country_data/'sentinel_4326')
    from water.basic_functions import printdf
    import os
    df = pd.read_csv(ppaths.country_data/'sentinel_4326_quality.csv')
    df = df.sort_values(by='percent', ascending=False)
    # printdf(df, 100)
    file_names_to_remove = df[df.percent>.05].file_name.unique().tolist()
    files_to_remove = [ppaths.country_data/f'sentinel_4326/{file_name}' for file_name in file_names_to_remove]
    # print(files_to_remove)
    print(len(files_to_remove))

    for file in files_to_remove:
        if file.exists():
            print(file)
            os.remove(file)
            print(file.exists())