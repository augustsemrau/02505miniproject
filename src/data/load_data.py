
# importing sys
import sys
# adding /features to the system path
sys.path.insert(0, '/home/augustsemrau/drive/M1semester/02506_AdvancedImageAnalysis/02505miniproject/')

from src.features.image_augmentation import elastic, add_noise, shear, rotate, zoom

import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor()])

import PIL.Image as Image
import os
import glob
import numpy as np
import cv2 


def transform(image, label, transform_type=0, train=True, validation=False):
    """
    Takes image, mask and desired transformation and returns transformed image and mask
    :param image:
    :param mask:
    :param transform_type:
    :return: Transformed images and masks
    """

    func = [elastic, add_noise, shear, rotate, zoom][transform_type]
    image, label = func(image, label)

    # Apply the transformations defined in the input transform parameter. Remember to end it with 'to_tensor'
    image = transform(image)
    label = transform(label)  # if (self._train or self._validation) else self._test_mask
    return image, label


def load_label(label_paths, idx):
    """
    Helper function to load in labels
    :param idx: index to return
    :return: label with the given index
    """
    label = Image.open(label_paths[idx])
    label = np.asarray(label).copy()
    label[label>0] = 255
    return label


def load_data(path, label_path, size, transform, train=True, validation=False):
    """
    Loads data from the given path and returns the data in the form of a dataset
    :param path:
    :param size:
    :param transform:
    :param train:
    :param validation:
    :return:
    """

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    mask = self._load_mask(idx) if self._train or self._validation else self._test_mask
    X, y = self.transform(image, mask)
    return X, y


def load_all_data(transform, train=True):
    """
    Loads all data from the given path and returns the data in the form of a dataset
    :param transform: Wished transformation(s)
    :return: Return a X, Y pair of image and mask
    """

    
    data_path = os.path.join('data/raw/EM_ISBI_Challenge/' 'train_images' if train else 'test_images')
    image_paths = sorted(glob.glob(os.path.join(data_path, "*.png")))
    label_paths = sorted(glob.glob(os.path.join('data/raw/EM_ISBI_Challenge/' 'train_labels', "*.png"))) if train else None
    
    for idx in range(len(image_paths)):
        image_path = image_paths[idx]
        load_data

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = self._load_mask(idx) if self._train or self._validation else self._test_mask
    X, y = self.transform(image, mask)
    return X, y
