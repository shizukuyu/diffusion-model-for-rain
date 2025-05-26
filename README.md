# diffusion-model-for-rain

## Data

I analyze hourly precipitation data over South China at a 1 km resolution (file: Z_SURF_C_BABJ_20220101001916_P_CMPA_RT_BCGZ_0P01_HOR-PRE.nc), a product from the China Meteorological Administration (CMA). Additionally, I use low-resolution ERA5 data at 25 km resolution, which includes variables such as *u*, *v*, *t*, rh, and geopotential at different pressure levels (580, 700, 500, and 300 hPa). These variables are expected to improve precipitation data generation.

Initially, both the high-resolution (CMA) and low-resolution (ERA5) data were stored in GRIB2 (or GRIB) format. They were first converted to NetCDF and later reshaped into .npy format for processing.

In this study, the high-resolution CMA data serves as the ground truth (label), while the low-resolution ERA5 data is interpolated to match the high-resolution grid and used as input. The diffusion model is trained to predict the distribution of the difference between the high-resolution data and the interpolated low-resolution data. For normalization, the mean and standard deviation of each variable are computed. Wind-related parameters (*u*, *v*) are scaled to the range [-1, 1], while the remaining variables are normalized to [0, 1].

## Modeling framework

This model is a generative diffusion model based on DDPM. The learned denoising network
is parameterized by a neural network with a UNet architecture. The modeling
framework used to design the model is `PyTorch`. Model architectures are put in /.model folder, such as Unet. The model was trained and
evaluated using NVIDIA H800 GPUs.
