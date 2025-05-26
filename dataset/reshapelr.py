import xarray as xr
import numpy as np
import os
from glob import glob

input_dir = "/workspace3/suwen/ddpm/dataset/2022_850_uv-rh-temp_025"
output_dir = "/workspace3/suwen/ddpm/dataset/lr"
os.makedirs(output_dir, exist_ok=True)

nc_files = glob(os.path.join(input_dir, "*.nc"))

for nc_file in nc_files:
    ds = xr.open_dataset(nc_file)
    data = np.stack([
        ds['r'].values,  # relative_humidity
        ds['t'].values,  # temperature
        ds['u'].values,  # u_wind
        ds['v'].values   # v_wind
    ], axis=0)  # shape: (lat, lon, 4)
    
    filename = os.path.basename(nc_file).replace(".nc", ".npy")
    output_path = os.path.join(output_dir, filename)
    np.save(output_path, data)
    print(f"Saved: {output_path}")

print("All files processed!")