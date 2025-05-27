import os 
import xarray as xr
import numpy as np
import xesmf as xe
from src.mjo.utils.RMM.eof import detrend_anomalies, latitude_band_average
from src.mjo.utils.RMM.io import save_rmm_indices

def main():

    #reference period to use   
    reference_start = '1979-09-07'
    reference_end = '2001-12-31'
    
    #slice data between start and end dates
    start_date = '1979-09-07'
    end_date = '2022-12-31'

    olr_file_path = '/glade/derecho/scratch/kvirji/DATA/NOAA/OLR/PSL_interpolated/olr.day.mean.nc'
    era5_file_path = '/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr'
    reference_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/EOF/reference_period_{reference_start}_to_{reference_end}'
    save_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/RMM/reference_period_{reference_start}_to_{reference_end}'

    os.makedirs(save_dir, exist_ok=True)

    #open required datasets
    raw_olr_ds = xr.open_dataset(olr_file_path).isel(time=slice(249, None)) #remove some corrupted dates
    raw_era5_ds = xr.open_zarr(era5_file_path)
    seasonal_cycle_ds = xr.open_dataset(os.path.join(reference_dir, 'seasonal_cycle.nc'))
    normalization_factor_ds = xr.open_dataset(os.path.join(reference_dir, 'normalization_factor.nc'))
    EOF_ds = xr.open_dataset(os.path.join(reference_dir, 'eof.nc'))

    olr_data = raw_olr_ds['olr'].to_dataset()
    u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850')
    u200_data = raw_era5_ds['u_component_of_wind'].sel(level=200, drop=True).to_dataset(name='u200')

    # regrid to 2.5°
    target_lat = np.arange(-90, 90, 2.5)
    target_lon = np.arange(0, 360, 2.5)
    olr_regridder = xe.Regridder(olr_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    u850_regridder = xe.Regridder(u850_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    u200_regridder = xe.Regridder(u200_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)

    olr_data_2p5d = olr_regridder(olr_data)
    u850_data_2p5d = u850_regridder(u850_data)
    u200_data_2p5d = u200_regridder(u200_data)

    #slide and load data for required period
    period = slice(start_date, end_date)
    olr_data_2p5d = olr_data_2p5d.sel(time=period).load()
    u850_data_2p5d = u850_data_2p5d.sel(time=period).load()
    u200_data_2p5d = u200_data_2p5d.sel(time=period).load()

    olr_seasonal_cycle = seasonal_cycle_ds['olr'].sel(dayofyear=olr_data_2p5d.time.dt.dayofyear)
    u850_seasonal_cycle = seasonal_cycle_ds['u850'].sel(dayofyear=u850_data_2p5d.time.dt.dayofyear)
    u200_seasonal_cycle = seasonal_cycle_ds['u200'].sel(dayofyear=u200_data_2p5d.time.dt.dayofyear)

    # subtract mean and fisrt three harmonics
    olr_anomalies = olr_data_2p5d - olr_seasonal_cycle
    u850_anomalies = u850_data_2p5d - u850_seasonal_cycle
    u200_anomalies = u200_data_2p5d - u200_seasonal_cycle

    #detrend anomalies by removing 120d running mean
    detrended_olr_anomalies = detrend_anomalies(olr_anomalies)
    detrended_u850_anomalies = detrend_anomalies(u850_anomalies)
    detrended_u200_anomalies = detrend_anomalies(u200_anomalies)

    #average over 15S-15N
    detrended_olr_anomalies_latitude_band_avg = latitude_band_average(detrended_olr_anomalies)
    detrended_u850_anomalies_latitude_band_avg = latitude_band_average(detrended_u850_anomalies)
    detrended_u200_anomalies_latitude_band_avg = latitude_band_average(detrended_u200_anomalies)

    #normalize each variable with factors calculated from reference period
    detrended_olr_anomalies_latitude_band_avg_norm = detrended_olr_anomalies_latitude_band_avg / normalization_factor_ds['olr']
    detrended_u850_anomalies_latitude_band_avg_norm = detrended_u850_anomalies_latitude_band_avg / normalization_factor_ds['u850']
    detrended_u200_anomalies_latitude_band_avg_norm = detrended_u200_anomalies_latitude_band_avg / normalization_factor_ds['u200']

    #drop missing values
    detrended_olr_anomalies_latitude_band_avg_norm = detrended_olr_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')
    detrended_u850_anomalies_latitude_band_avg_norm = detrended_u850_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')
    detrended_u200_anomalies_latitude_band_avg_norm = detrended_u200_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')

    #ensure timesteps are aligned
    olr, u850, u200 = xr.align(
        detrended_olr_anomalies_latitude_band_avg_norm,
        detrended_u850_anomalies_latitude_band_avg_norm,
        detrended_u200_anomalies_latitude_band_avg_norm,
        join='inner'
    )

    #combine data
    X = xr.concat([olr['olr'], u850['u850'], u200['u200']], dim='lon')  # (time, 3 × lon)  

    #project data onto reference EOFs
    RMM1 = X.values @ EOF_ds['EOF1'].values
    RMM2 = X.values @ EOF_ds['EOF2'].values

    #divive RMM indices by factors calculated from reference period
    RMM1_norm = RMM1 / normalization_factor_ds['RMM1_std'].values
    RMM2_norm = RMM2 / normalization_factor_ds['RMM2_std'].values

    #save indices to txt file
    save_rmm_indices(
        time=olr.time.values,
        RMM1=RMM1_norm,
        RMM2=RMM2_norm,
        output_path=os.path.join(save_dir, 'rmm.txt')
    )


if __name__ == "__main__":
    main()
