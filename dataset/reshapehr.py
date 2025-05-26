import xarray as xr
import numpy as np
import os
from glob import glob

# 输入和输出路径
input_dir = "/workspace3/suwen/ddpm/dataset/2022-22"
output_dir = "/workspace3/suwen/ddpm/dataset/hr"
os.makedirs(output_dir, exist_ok=True)

# 获取所有匹配的NC文件（根据实际文件名调整模式）
nc_files = glob(os.path.join(input_dir, "Z_SURF_C_BABJ_*.nc"))

for nc_file in nc_files:
    # 读取NetCDF文件
    ds = xr.open_dataset(nc_file)
    
    # 提取唯一的变量（名为'unknown'）
    if 'unknown' in ds:
        data = ds['unknown'].values  # shape: (lat, lon)
    else:
        # 如果变量名不是'unknown'，尝试自动获取第一个变量
        var_name = list(ds.data_vars.keys())[0]
        data = ds[var_name].values
    
    # 添加通道维度 [height, width, 1]（符合DDPM输入格式）
    data = data[np.newaxis, ...]  # 或 np.expand_dims(data, axis=-1)
    
    # 生成输出路径（相同文件名，改为.npy后缀）
    filename = os.path.basename(nc_file).replace(".nc", ".npy")
    output_path = os.path.join(output_dir, filename)
    
    # 保存为.npy文件
    np.save(output_path, data)
    print(f"Processed: {filename} -> Shape: {data.shape}")

print("All files processed!")