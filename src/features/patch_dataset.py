import os
from torch.utils.data import Dataset

class TrainPatchData(Dataset):
    def __init__(self, patch_size):
        base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'processed')
        self.input_files = sorted(os.listdir(os.path.join(base_path, f'{patch_size}_train_patches')))
        self.target_files = sorted(os.listdir(os.path.join(base_path, f'{patch_size}_label_patches')))
        
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, i):
        return self.input_files[i], self.target_files[i]