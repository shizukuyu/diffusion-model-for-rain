import xarray as xr
import numpy as np
import os
from glob import glob

# 输入和输出路径
input_dir = "/workspace3/suwen/ddpm/dataset/2022_850_uv-rh-temp_025"
output_dir = "/workspace3/suwen/ddpm/dataset/lr"
os.makedirs(output_dir, exist_ok=True)

# 获取所有NC文件
nc_files = glob(os.path.join(input_dir, "*.nc"))

for nc_file in nc_files:
    # 读取NetCDF文件
    ds = xr.open_dataset(nc_file)
    
    # 提取四个变量并堆叠成多通道数组 [height, width, channels]
    data = np.stack([
        ds['r'].values,  # relative_humidity
        ds['t'].values,  # temperature
        ds['u'].values,  # u_wind
        ds['v'].values   # v_wind
    ], axis=0)  # shape: (lat, lon, 4)
    
    # 生成输出路径（相同文件名，但改为.npy后缀）
    filename = os.path.basename(nc_file).replace(".nc", ".npy")
    output_path = os.path.join(output_dir, filename)
    
    # 保存为.npy文件
    np.save(output_path, data)
    print(f"Saved: {output_path}")

print("All files processed!")