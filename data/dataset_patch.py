import numpy as np
import torch
from bisect import bisect
import random

class SR3_Dataset_patch(torch.utils.data.Dataset):
    def __init__(self, hr_paths, lr_paths, var, patch_size):
        self.target_hr = [np.load(path, mmap_mode='r+') for path in hr_paths]
        self.target_lr = [np.load(path, mmap_mode='r+') for path in lr_paths]
            
        self.target_hr = np.nan_to_num(self.target_hr, nan=0.0) 
        self.target_lr = np.nan_to_num(self.target_lr, nan=0.0)
    
        
        self.patch_size = patch_size
        self.data_count = len(hr_paths)  
        
        assert len(self.target_hr) == len(self.target_lr), "HR and LR counts not match"
        
        self.max = torch.from_numpy(np.load("/workspace3/suwen/ddpm/dataset/lr/max_values.npy")).float()
        self.min = torch.from_numpy(np.load("/workspace3/suwen/ddpm/dataset/lr/min_values.npy")).float()
        
        if self.max.ndim == 1:
            self.max = self.max.view(4, 1, 1)
        if self.min.ndim == 1:
            self.min = self.min.view(4, 1, 1)
    
    def normalize_data(self, data, var_index):
        max_ = self.max[var_index].view(1, 1, 1)
        min_ = self.min[var_index].view(1, 1, 1)
        
        if var_index in [0, 1]:  # r and t to [0,1]
            normalized = (data - min_) / (max_ - min_ + 1e-6)
        else:  # u and v to[-1,1]
            # print("max",max_)
            # print("min",min_)
            normalized = 2 * (data - min_) / (max_ - min_ + 1e-6) - 1
        return normalized
    
    def get_patch(self, hr, lr_inter):
        _, ih, iw = hr.shape
        ip = self.patch_size
        
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        
        hr_patch = hr[:, iy:iy+ip, ix:ix+ip]
        
        lr_patches = []
        for var_idx in range(4):
            lr_channel = lr_inter[var_idx:var_idx+1, iy:iy+ip, ix:ix+ip]
            lr_normalized = self.normalize_data(lr_channel, var_idx)
            lr_patches.append(lr_normalized)
        
        lr_patches = torch.cat(lr_patches, dim=0)
        
        return {
            "HR": hr_patch,
            "INTERPOLATED": lr_patches
        }
    
    def __len__(self):
        return self.data_count
    
    def __getitem__(self, index):
        hr = self.target_hr[index][:]  # HR shape [1,H,W]
        lr = self.target_lr[index][:]  # LR shape [4,H',W']
        
        hr_tensor = torch.from_numpy(hr).float()
        if hr_tensor.ndim == 2:
            hr_tensor = hr_tensor.unsqueeze(0) 
        
        lr_tensor = torch.from_numpy(lr).float()
        if lr_tensor.ndim == 2:
            lr_tensor = lr_tensor.unsqueeze(0)
        
        lr_inter = torch.nn.functional.interpolate(
            lr_tensor.unsqueeze(0),  # [1,4,H',W']
            size=hr_tensor.shape[-2:], 
            mode='bilinear'
        ).squeeze(0)  # [4,H,W]
        
        return self.get_patch(hr_tensor, lr_inter)

if __name__ == "__main__":
    lr_paths = ["/workspace3/suwen/ddpm/dataset/lr/ERA5_20220101_0000_005.npy"]
    hr_paths = ["/workspace3/suwen/ddpm/dataset/hr/Z_SURF_C_BABJ_20220101001916_P_CMPA_RT_BCGZ_0P01_HOR-PRE-2022010100_005.npy"]
    
    dataset = SR3_Dataset_patch(hr_paths, lr_paths, var=None, patch_size=128)
    sample = dataset[0]
    
    print(f"HR shape: {sample['HR'].shape}")      
    print(f"INTERPOLATED shape: {sample['INTERPOLATED'].shape}")  
    
    print("R channel range:", sample['INTERPOLATED'][0].min(), sample['INTERPOLATED'][0].max())  # [0,1]
    print("U channel range:", sample['INTERPOLATED'][2].min(), sample['INTERPOLATED'][2].max())  # [-1,1]