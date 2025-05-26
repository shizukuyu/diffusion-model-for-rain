import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class DDPMPreprocessor:
    def __init__(self, hr_path, lr_path, lr_max_path, lr_min_path):
        """
        :param hr_path
        :param lr_path
        :param lr_max_path
        :param lr_min_path
        """
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.lr_max_path = lr_max_path
        self.lr_min_path = lr_min_path
        self.hr_data = None
        self.lr_data = None
        self.processed_hr = None
        self.processed_lr = None
        
    def load_data(self):
        try:
            self.hr_data = np.load(self.hr_path)
            self.lr_data = np.load(self.lr_path)
            print(f"HR shape: {self.hr_data.shape}")
            print(f"LR shape: {self.lr_data.shape}")
        except Exception as e:
            print(f"Error: {e}")
            return False
        return True
    
    def normalize_lr(self):
        if self.lr_data is None:
            print("load data first")
            return False
            
        try:
            max_values = np.load(self.lr_max_path)
            min_values = np.load(self.lr_min_path)
            
            for channel in range(self.lr_data.shape[0]):
                self.lr_data[channel] = (self.lr_data[channel] - min_values[channel]) / (max_values[channel] - min_values[channel] + 1e-8)
                self.lr_data[channel] = np.clip(self.lr_data[channel], 0, 1)
                
            print("LR normalization finished")
            return True
        except Exception as e:
            print(f"LR normalization error: {e}")
            return False
    
    def adjust_dimensions(self):
        if self.hr_data is None or self.lr_data is None:
            print("load data first")
            return False
            
        hr_height, hr_width = self.hr_data.shape[1], self.hr_data.shape[2]
        new_hr_height = hr_height if hr_height % 2 == 0 else hr_height + 1
        new_hr_width = hr_width if hr_width % 2 == 0 else hr_width + 1
        
        hr_tensor = torch.tensor(self.hr_data, dtype=torch.float32).unsqueeze(0)  
        hr_tensor = F.interpolate(hr_tensor, size=(new_hr_height, new_hr_width), mode='bilinear', align_corners=False)
        self.processed_hr = hr_tensor.squeeze(0).numpy()  
        
        lr_height, lr_width = self.lr_data.shape[1], self.lr_data.shape[2]
        new_lr_height = lr_height if lr_height % 2 == 0 else lr_height + 1
        new_lr_width = lr_width if lr_width % 2 == 0 else lr_width + 1
        
        lr_tensor = torch.tensor(self.lr_data, dtype=torch.float32).unsqueeze(0)  
        lr_tensor = F.interpolate(lr_tensor, size=(new_lr_height, new_lr_width), mode='bilinear', align_corners=False)
        self.processed_lr = lr_tensor.squeeze(0).numpy()  
        
        print(f"HR shape adjust: {self.hr_data.shape} -> {self.processed_hr.shape}")
        print(f"LR shape adjust: {self.lr_data.shape} -> {self.processed_lr.shape}")
        return True
    
    def save_processed_data(self, output_dir='/workspace3/suwen/ddpm/dataset/processed_data'):
        if self.processed_hr is None or self.processed_lr is None:
            print("load data first")
            return False
            
        os.makedirs(output_dir, exist_ok=True)
        
        hr_output_path = os.path.join(output_dir, 'processed_hr.npy')
        lr_output_path = os.path.join(output_dir, 'processed_lr.npy')
        
        np.save(hr_output_path, self.processed_hr)
        np.save(lr_output_path, self.processed_lr)
        
        print(f"data saved to: {output_dir}")
        return True

class DDPMDataset(Dataset):
    def __init__(self, hr_data, lr_data):
        """
        :param hr_data
        :param lr_data:
        """
        self.hr_data = hr_data
        self.lr_data = lr_data
        
    def __len__(self):
        return 1  
        
    def __getitem__(self, idx):
        hr_tensor = torch.tensor(self.hr_data, dtype=torch.float32)
        lr_tensor = torch.tensor(self.lr_data, dtype=torch.float32)
        
        return {'hr': hr_tensor, 'lr': lr_tensor}

def create_data_loader(hr_data, lr_data, batch_size=1):
    dataset = DDPMDataset(hr_data, lr_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    hr_path = '/workspace3/suwen/ddpm/dataset/hr/Z_SURF_C_BABJ_20220101001916_P_CMPA_RT_BCGZ_0P01_HOR-PRE-2022010100_005.npy'
    lr_path = '/workspace3/suwen/ddpm/dataset/lr/ERA5_20220101_0000_005.npy'
    lr_max_path = '/workspace3/suwen/ddpm/dataset/lr/max_values.npy'
    lr_min_path = '/workspace3/suwen/ddpm/dataset/lr/min_values.npy'

    processor = DDPMPreprocessor(hr_path, lr_path, lr_max_path, lr_min_path)
    

    if processor.load_data() and processor.normalize_lr() and processor.adjust_dimensions():
        processor.save_processed_data()
        
        data_loader = create_data_loader(processor.processed_hr, processor.processed_lr)
        print("data loader sucess")
        
        for batch in data_loader:
            print(f"bacth HR data shape: {batch['hr'].shape}")
            print(f"bacth LR data shape: {batch['lr'].shape}")
            break
    else:
        print("data process failed")    