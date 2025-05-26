# diffusion-model-for-rain

## Data

I consider precipitation data at hourly resolution over the South China, at
1 km resolution (`Z_SURF_C_BABJ_20220101001916_P_CMPA_RT_BCGZ_0P01_HOR-PRE.nc`) which is the product from China Meteorological Administration.
And the low-resolution data from ERA5, at 25 km resolution, variables contain u,v,t,rh,geopotential from different level (580, 700, 500, 300) which would be benefical to the generation of precipitation data.

The original high-resolution data and low-resolution data is in the format of GRB2 or GRIB, in the beggining the data is converted to netcdf and finnaly reshape into .npy format.

I use high-resolution data from CMA, which serves as the label, and low-resolution data interpolated to the high-resolution
grid, which serves as the input.This diffusion models are trained to predict samples of the distribution of the
difference between the high-resolution data and the interpolated low-resolution
data as the target. For normalization, we also compute the mean and standard
deviation of each variable, wind parameters in the range of [-1, 1], the left parameters are normalized into [0, 1].

## Modeling framework

This model is a generative diffusion model based on DDPM. The learned denoising network
is parameterized by a neural network with a UNet architecture. The modeling
framework used to design the model is `PyTorch`. Model architectures are put in /.model folder, such as Unet. The model was trained and
evaluated using NVIDIA H800 GPUs.
