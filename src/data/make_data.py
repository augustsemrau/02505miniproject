import os
from tqdm import tqdm
import requests as req
from zipfile import ZipFile 

IMG_DIR = 'EM_ISBI_Challenge'
ZIP_FILE_NAME = f'{IMG_DIR}.zip'
FILE_URI = f'http://www2.imm.dtu.dk/courses/02506/data/{ZIP_FILE_NAME}'
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ZIP_FILE_PATH = os.path.join(BASE_PATH, '..', '..', 'data', 'raw', ZIP_FILE_NAME)
EXTRACT_PATH = os.path.join(BASE_PATH, '..', '..', 'data', 'raw')

def get_data():
    print('Downloading EM ISBI Challange data...')
    r = req.get(FILE_URI, stream=True)

    print(f'Saving face data to {ZIP_FILE_PATH}')
    total_size_in_bytes= int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(ZIP_FILE_PATH, 'wb') as faces:
        for data in r.iter_content(block_size):
            progress_bar.update(len(data))
            faces.write(data)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    print('Unzipping data...')
    with ZipFile(ZIP_FILE_PATH, 'r') as zip:
        zip.extractall(EXTRACT_PATH)

    os.remove(ZIP_FILE_PATH)

    print('Data created sucessfully!')

if __name__ == '__main__':
    data_dir = os.path.join(EXTRACT_PATH, IMG_DIR)
    if not os.path.exists(data_dir):
        get_data()
    else:
        print(f'Data dir {data_dir} already exsits')