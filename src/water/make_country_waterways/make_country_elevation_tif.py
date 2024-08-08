from water.make_country_waterways.cut_data import cut_data_to_match_file_list
from water.basic_functions import ppaths, Path, tt, time_elapsed


def make_country_elevation_tif(
        save_dir_path: Path,
        output_dir: Path,
        elevation_path: Path=ppaths.country_data/'elevation'):
    cut_data_to_match_file_list(
        save_dir=save_dir_path/'elevation_merged', num_proc=1,
        file_paths=list(output_dir.glob('*.tif')), data_dir=elevation_path
    )


if __name__ == '__main__':
    country = 'rwanda'
    make_country_elevation_tif(
        save_dir_path=ppaths.country_data/'africa/rwanda', output_dir=ppaths.country_data/'africa/rwanda/output_data_merged'
    )
    # countries = [
    #     'germany', 'france', 'belgium', 'netherlands', 'switzerland', 'austria', 'italy', 'spain',
    #     'burundi', 'liberia', 'panama', 'nicaragua', 'bolivia', 'zambia', 'ethiopia',  'somalia', 'south_sudan',
    #     'rwanda', 'uganda', 'ivory_coast', 'kenya', 'tanzania'
    # ]
    # for country in countries:
    #     print(country)
    #     s = tt()
    #     make_country_elevation_tif(country=country)
    #     time_elapsed(s, 2)