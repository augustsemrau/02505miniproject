import torch
import glob 
import os
import cv2
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.features.image_augmentation import elastic, add_noise, shear, rotate, zoom

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Dataset(torch.utils.data.Dataset):
    '''  Dataset which loads all images for training or testing'''
    def __init__(self, train, val=False, augmentation_count=None):
        self._train = train
        self.augmentation_count = augmentation_count
        self._transform = transforms.Compose([transforms.ToTensor()])

        if self._train:
            image_dir = "data/raw/EM_ISBI_Challenge/train_images"
            label_dir = "data/raw/EM_ISBI_Challenge/train_labels"
        else:
            image_dir = "data/raw/EM_ISBI_Challenge/test_images"

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.png"))) if self._train else None

        self.images = []
        self.labels = []
        
        if train and not val:
            r = range(5,len(self.image_paths))
        elif train and val:
            r = range(0,5)
        else:
            r = range(len(self.image_paths))

        for idx in r:
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)/255
            self.images.append(torch.tensor(image).unsqueeze(0))#, dtype=torch.float64).unsqueeze(0))
            if self._train:
                label = np.asarray(Image.open(self.label_paths[idx])).copy()
                label[label>0] = 255/255
                self.labels.append(torch.tensor(label).unsqueeze(0))#, dtype=torch.float64).unsqueeze(0))
            else:
                self.labels.append(torch.zeros([512,512]).unsqueeze(0))#, dtype=torch.float64).unsqueeze(0))

        

        if self.augmentation_count is not None and train:
            for idx in r:
                for aug in range(augmentation_count):
                    func = np.random.choice([rotate, zoom])#add_noise, shear, zoom])#,elastic])
                    image, label = func(self.images[idx].squeeze(0), self.labels[idx].squeeze(0))

                    # Apply the transformations defined in the input transform parameter. Remember to end it with 'to_tensor'
                    self.images.append(torch.tensor(self._transform(image)))#, dtype=torch.float64))
                    self.labels.append(torch.tensor(self._transform(label)))#, dtype=torch.float64))


    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.images)


class DatasetPatch(torch.utils.data.Dataset):
    '''  Dataset which loads all images for training or testing'''
    def __init__(self, train, val=False, size=512):
        self._train = train
        self._size = size
        self._transform = transforms.Compose([transforms.ToTensor()])
        
        if self._train:
            image_dir = f"data/processed/{self._size}_train_patches"
            label_dir = f"data/processed/{self._size}_label_patches"
        else:
            image_dir = f"data/processed/{self._size}_test_patches"

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.png"))) if self._train else None

        self.images = []
        self.labels = []
        
        set_seed(seed=0)
        r_v = random.sample(range(len(self.image_paths)), int(len(self.image_paths)*0.2))
        r_t = [i for i in range(len(self.image_paths)) if i not in r_v]

        if train and not val:
            r = r_t
        elif train and val:
            r = r_v
        else:
            r = range(len(self.image_paths))

        for idx in r:
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)/255
            self.images.append(torch.tensor(image).unsqueeze(0))#, dtype=torch.float64).unsqueeze(0))
            if self._train:
                label = np.asarray(Image.open(self.label_paths[idx])).copy()
                label[label>0] = 255/255
                self.labels.append(torch.tensor(label).unsqueeze(0))#, dtype=torch.float64).unsqueeze(0))
            else:
                self.labels.append(torch.zeros([self._size,self._size]).unsqueeze(0))#, dtype=torch.float64).unsqueeze(0))

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.images)