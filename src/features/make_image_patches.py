import os
from tqdm import tqdm
from torchvision import io, transforms
from torchvision.transforms.functional import to_pil_image

IMG_SIZE = 256
PATCH_SIZE = 128
FILE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_SAVE_DIR = os.path.join(FILE_DIR, '..', '..', 'data', 'processed', 'train_patches')
TRAIN_DIR = os.path.join(FILE_DIR, '..', '..', 'data', 'raw', 'EM_ISBI_Challenge', 'train_images')
TEST_SAVE_DIR = os.path.join(FILE_DIR, '..', '..', 'data', 'processed', 'test_patches')
TEST_DIR = os.path.join(FILE_DIR, '..', '..', 'data', 'raw', 'EM_ISBI_Challenge', 'test_images')
LABEL_SAVE_DIR = os.path.join(FILE_DIR, '..', '..', 'data', 'processed', 'label_patches')
LABEL_DIR = os.path.join(FILE_DIR, '..', '..', 'data', 'raw', 'EM_ISBI_Challenge', 'train_labels')


def try_make_patch_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    else:
        raise Exception(f'''Folder {folder_path} already exists!\n
                         If you want to continue remove the folder and run this script again.''')


def create_patches(img_dir, save_dir):
    '''
    Loads all images in the given image directory and splits them into patches
    and then saves them to the given save directory.

    :param img_files:   Path to directory, where all images will be read from 
    :param save_dir:    Path to directory in which to save the image patches
    '''
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    img_files = sorted([file for file in os.listdir(img_dir) if file.endswith('.png')])

    for img_file in tqdm(img_files):
        img = resize(io.read_image(os.path.join(img_dir, img_file)))
        patches = img.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)

        patch_num = 1
        for i in range(patches.shape[1]):
            for j in range(patches.shape[2]):
                patch = patches[:,i,j]
                pil_img = to_pil_image(patch)
                pil_img.save(os.path.join(save_dir, f'{img_file}_{patch_num}.png'))
                patch_num += 1



if __name__ == '__main__':
    print(f'Creating training image patches of size {PATCH_SIZE} X {PATCH_SIZE} pixels')
    try_make_patch_folder(TRAIN_SAVE_DIR)
    create_patches(TRAIN_DIR, TRAIN_SAVE_DIR)
    print(f'Training patches saved to {TRAIN_DIR}')

    print(f'Creating test image patches of size {PATCH_SIZE} X {PATCH_SIZE} pixels')
    try_make_patch_folder(TEST_SAVE_DIR)
    create_patches(TEST_DIR, TEST_SAVE_DIR)
    print(f'Training patches saved to {TEST_DIR}')

    print(f'Creating label image patches of size {PATCH_SIZE} X {PATCH_SIZE} pixels')
    try_make_patch_folder(LABEL_SAVE_DIR)
    create_patches(LABEL_DIR, LABEL_SAVE_DIR)
    print(f'Label patches saved to {LABEL_DIR}')

    print('Sucessfully create training and label image patches!')