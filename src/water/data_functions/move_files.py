from water.basic_functions import ppaths
from pathlib import Path
import os
def move_files(base_dir:Path):
    hu4_dirs = base_dir.glob('hu4_*')
    for hu4_dir in hu4_dirs:
        hu4_files = hu4_dir.glob('*')
        for file in hu4_files:
            file_name = file.name
            new_name = base_dir/file_name
            if not new_name.exists():
                file.rename(new_name)
            else:
                print(new_name)



def copy_ts_files_to_og():
    burned_ts_files = list((ppaths.waterway/'waterways_ts_burned').glob('*'))
    sentinel_ts_dir = ppaths.waterway/'sentinel_ts_4326'
    burned_dir = ppaths.waterway/'waterways_burned'
    sentinel_dir = ppaths.waterway/'sentinel_4326'
    num_moved = 0
    for file in burned_ts_files:
        file_name = file.name
        sen_ts_name = file.name.split('.tif')[0]
        if not (burned_dir/file_name).exists():
            num_moved += 1
            sentinel_file = sentinel_ts_dir/sen_ts_name/'1.tif'
            new_sentinel_path = sentinel_dir/file_name
            new_burned_path = burned_dir/file_name
            os.system(f'cp {file} {new_burned_path}')
            os.system(f'cp {sentinel_file} {new_sentinel_path}')
            # print(new_sentinel_path, new_burned_path)
            # break
    print(num_moved)


if __name__ == '__main__':
    copy_ts_files_to_og()
    # move_files(ppaths.waterway/'sentinel_ts_4326')