from importlib.resources import path
import os
from pydoc import describe
from requests import patch
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision import io, transforms
from torchvision.transforms.functional import to_pil_image

IMG_SIZE = 256
FILE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(FILE_DIR, '..', '..', 'data', 'raw', 'EM_ISBI_Challenge', 'train_images')
TEST_DIR = os.path.join(FILE_DIR, '..', '..', 'data', 'raw', 'EM_ISBI_Challenge', 'test_images')
LABEL_DIR = os.path.join(FILE_DIR, '..', '..', 'data', 'raw', 'EM_ISBI_Challenge', 'train_labels')

def get_train_patch_dir(patch_size):
    return os.path.join(FILE_DIR, '..', '..', 'data', 'processed', f'{patch_size}_train_patches')

def get_test_patch_dir(patch_size):
    return os.path.join(FILE_DIR, '..', '..', 'data', 'processed', f'{patch_size}_test_patches')

def get_label_patch_dir(patch_size):
    return os.path.join(FILE_DIR, '..', '..', 'data', 'processed', f'{patch_size}_label_patches')

def try_make_patch_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        return True
    else:
        print('Patches already exist')
        return False


def create_patches(img_dir, save_dir, patch_size):
    '''
    Loads all images in the given image directory and splits them into patches
    and then saves them to the given save directory.

    :param img_files:   Path to directory, where all images will be read from 
    :param save_dir:    Path to directory in which to save the image patches
    :param patch_size:  Size of the (patch_size X patch_size) patch which the image should be split into
    '''
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    img_files = sorted([file for file in os.listdir(img_dir) if file.endswith('.png')])

    for img_file in tqdm(img_files):
        img = resize(io.read_image(os.path.join(img_dir, img_file)))
        patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

        patch_num = 1
        for i in range(patches.shape[1]):
            for j in range(patches.shape[2]):
                patch = patches[:,i,j]
                pil_img = to_pil_image(patch)
                pil_img.save(os.path.join(save_dir, f'{img_file}_{patch_num}.png'))
                patch_num += 1



if __name__ == '__main__':
    arg_parser = ArgumentParser(description='Splits images into patches and saves them in data/processed')

    arg_parser.add_argument('--patch_size', type=int, help='Width and height of quadratic image patch', required=True)

    args = arg_parser.parse_args()

    patch_size = args.patch_size

    print(f'Creating training image patches of size {patch_size} X {patch_size} pixels')
    train_save_dir = get_train_patch_dir(patch_size)
    if try_make_patch_folder(train_save_dir):
        create_patches(TRAIN_DIR, train_save_dir, patch_size)
        print(f'Training patches saved to {TRAIN_DIR}')

    print(f'Creating test image patches of size {patch_size} X {patch_size} pixels')
    test_save_dir = get_test_patch_dir(patch_size)
    if try_make_patch_folder(test_save_dir):
        create_patches(TEST_DIR, test_save_dir, patch_size)
        print(f'Training patches saved to {TEST_DIR}')

    print(f'Creating label image patches of size {patch_size} X {patch_size} pixels')
    label_save_dir = get_label_patch_dir(patch_size)
    if try_make_patch_folder(label_save_dir):
        create_patches(LABEL_DIR, label_save_dir, patch_size)
        print(f'Label patches saved to {LABEL_DIR}')

    print('Sucessfully create training and label image patches!')