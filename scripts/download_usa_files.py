from water.basic_functions import ppaths, multi_download, printdf, Path
import pandas as pd

# usa_path = ppaths.data/'usa'
# download_text_files = usa_path/'download_text_files'


def extract_dl_links(text_file):
    with open(text_file, 'r') as f:
        text = f.read()
    dl_links = text.split('\n')
    return dl_links


def make_save_path(dir_path, dl_url):
    file_name = Path(dl_url).name
    save_path = dir_path/file_name
    return save_path


def main():
    usa_path = ppaths.waterway/'waterway_storage/usa_data'
    download_text_files = ppaths.waterway/'download_text_files'
    txt_files = list(download_text_files.glob('*.txt'))
    download_urls = []
    save_as = []
    for file in txt_files:
        possible_urls = extract_dl_links(file)
        if len(possible_urls) < 3400:
            dir_name = file.name.split('.')[0]
            dir_path = usa_path/dir_name
            # dir_path = Path('/media/matthew/Behemoth/data/usa_waterway_data')/dir_name
            if not dir_path.exists():
                dir_path.mkdir()
            possible_sa = [make_save_path(dir_path, url) for url in possible_urls]
            for (url, sa) in zip(possible_urls, possible_sa):
                if not sa.exists():
                    download_urls.append(url)
                    save_as.append(sa)
    multi_download(
        url_list=download_urls,
        save_path_list=save_as,
        extract_zip_files=False,
        delete_zip_files_after_extraction=False,
        num_proc=22
    )

if __name__ == '__main__':
    main()