import os
import torch
from torchvision import io
from torchvision.transforms.functional import to_pil_image

def merge_patches(patches, patch_size, image_size):
    '''
    Merges a tensor of image patches back together.

    :param patches:     List of image patches to merge back together
    :param patch_size:  The size of the patches i.e. an integer for the shape (patch_size X patch_size)
    :param image_size:  The width or height of the image to output

    :returns:           A single image created from merging the patches back together
    '''
    N = int(image_size/patch_size)

    # Merge first row of patches back together
    img_tensor = torch.cat(patches[0:N], dim=2)

    # Merge remaining rows of patches back together
    for i in range(1, N):
        row_n = torch.cat(patches[i*N:(i+1)*N], dim=2)
        img_tensor = torch.cat([img_tensor, row_n], dim=1)

    return img_tensor


if __name__ == '__main__':
    # For testing
    base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data', 'processed', '256_train_patches')

    img1 = io.read_image(os.path.join(base_path, 'train_01_1.png'))
    img2 = io.read_image(os.path.join(base_path, 'train_01_2.png'))
    img3 = io.read_image(os.path.join(base_path, 'train_01_3.png'))
    img4 = io.read_image(os.path.join(base_path, 'train_01_4.png'))

    merged_img = merge_patches([img1, img2, img3, img4], 256, 512)
    img = to_pil_image(merged_img)
    img.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test.png'))