import xarray as xr
import numpy as np
import os
from glob import glob

input_dir = "/workspace3/suwen/ddpm/dataset/2022-22"
output_dir = "/workspace3/suwen/ddpm/dataset/hr"
os.makedirs(output_dir, exist_ok=True)

nc_files = glob(os.path.join(input_dir, "Z_SURF_C_BABJ_*.nc"))

for nc_file in nc_files:
    ds = xr.open_dataset(nc_file)
    
    if 'unknown' in ds:
        data = ds['unknown'].values  # shape: (lat, lon)
    else:
        var_name = list(ds.data_vars.keys())[0]
        data = ds[var_name].values
    data = data[np.newaxis, ...] 
    
    filename = os.path.basename(nc_file).replace(".nc", ".npy")
    output_path = os.path.join(output_dir, filename)
    
    np.save(output_path, data)
    print(f"Processed: {filename} -> Shape: {data.shape}")

print("All files processed!")