import os 
import glob
import warnings
import xarray as xr
import numpy as np
import xesmf as xe
from tqdm import tqdm
from mjo.utils.RMM.eof import detrend_anomalies, latitude_band_average
from mjo.utils.RMM.io import save_rmm_indices
from mjo.utils.RMM.FuXi.utils import format, walk_to_forecast_dir

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS.*")

def compute_ensembe_mean(forecast_root_dir):
    # Open (lazy)
    ds_ttr = xr.open_dataset(os.path.join(forecast_root_dir, 'ttr', 'ttr.nc'))
    ds_u   = xr.open_dataset(os.path.join(forecast_root_dir, 'u', 'u.nc'))

    # Ensemble means over the 'number' dimension
    ttr_mean = ds_ttr['ttr'].mean('number', keep_attrs=True)
    u_mean   = ds_u['u'].mean('number', keep_attrs=True)

    # Wrap each in its own Dataset so coords/attrs carry through cleanly
    ttr_out = xr.Dataset({'ttr': ttr_mean})
    u_out   = xr.Dataset({'u': u_mean})
   
    # File paths
    ttr_path = os.path.join(forecast_root_dir, 'ttr', 'ttr_ensmean.nc')
    u_path   = os.path.join(forecast_root_dir, 'u',   'u_ensmean.nc')

    # Write
    ttr_out.to_netcdf(ttr_path)
    u_out.to_netcdf(u_path)

    # Close inputs (optional but tidy)
    ds_ttr.close(); ds_u.close()

    print(f"Saved:\n  {ttr_path}\n  {u_path}")
    return 


def main():
    
    olr_file_path = '/glade/derecho/scratch/kvirji/DATA/NOAA/OLR/PSL_interpolated/olr.day.mean.nc'
    era5_file_path = '/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr'
    reference_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/U200/EOF'
    save_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/U200/ECMWF'

    os.makedirs(save_dir, exist_ok=True)

    #open required datasets
    raw_olr_ds = xr.open_dataset(olr_file_path).isel(time=slice(249, None)) #remove some corrupted dates
    raw_era5_ds = xr.open_zarr(era5_file_path)
    seasonal_cycle_ds = xr.open_dataset(os.path.join(reference_dir, 'seasonal_cycle.nc')).load()
    normalization_factor_ds = xr.open_dataset(os.path.join(reference_dir, 'normalization_factor.nc')).load()
    EOF_ds = xr.open_dataset(os.path.join(reference_dir, 'eof.nc')).load()

    gt_olr_data = raw_olr_ds['olr'].to_dataset().load()
    gt_u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850').load()
    gt_u200_data = raw_era5_ds['u_component_of_wind'].sel(level=200, drop=True).to_dataset(name='u200').load()

    raw_olr_ds.close()
    raw_era5_ds.close()

    target_lat = np.arange(-90, 90, 2.5)
    target_lon = np.arange(0, 360, 2.5)
    gt_olr_regridder = xe.Regridder(gt_olr_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    gt_u850_regridder = xe.Regridder(gt_u850_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    gt_u200_regridder = xe.Regridder(gt_u200_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    gt_olr_data_2p5d = gt_olr_regridder(gt_olr_data)
    gt_u850_data_2p5d = gt_u850_regridder(gt_u850_data)
    gt_u200_data_2p5d = gt_u200_regridder(gt_u200_data)

    forecast_root_dir = '/glade/derecho/scratch/kvirji/s2s.dir/ecmwf/'
    init_date = '2017-02-01'
    #compute_ensembe_mean(forecast_root_dir)
    ds_ttr = xr.open_dataset(os.path.join(forecast_root_dir, 'ttr', 'ttr_ensmean.nc')).rename_dims(latitude='lat', longitude='lon')
    ds_u   = xr.open_dataset(os.path.join(forecast_root_dir, 'u', 'u_ensmean.nc')).rename_dims(latitude='lat', longitude='lon')

    ttr = ds_ttr['ttr'].sel(forecast_reference_time=init_date).to_dataset(name='ttr')
    u200 = ds_u['u'].sel(pressure_level=200, forecast_reference_time=init_date).to_dataset(name='u200').drop_vars('pressure_level')
    u850 = ds_u['u'].sel(pressure_level=850, forecast_reference_time=init_date).to_dataset(name='u850').drop_vars('pressure_level')

    # regrid to 2.5°
    forecast_ttr_regridder = xe.Regridder(ttr, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    forecast_u200_regridder = xe.Regridder(u200, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    forecast_u850_regridder = xe.Regridder(u850, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)

    ttr_2p5d = forecast_ttr_regridder(ttr)
    u200_2p5d = forecast_u200_regridder(u200)
    u850_2p5d = forecast_u850_regridder(u850)

    #slide ground truth data for required period
    gt_end_date = np.datetime64(init_date)
    gt_start_date = gt_end_date - np.timedelta64(118, 'D')
    period = slice(gt_start_date, gt_end_date)
    gt_olr_data_2p5d_slice = gt_olr_data_2p5d.sel(time=period)
    gt_u850_data_2p5d_slice = gt_u850_data_2p5d.sel(time=period)
    gt_u200_data_2p5d_slice  = gt_u200_data_2p5d.sel(time=period)


    # Day 1: keep the first forecast step "as is" (then convert units/sign)
    olr_day1 = (-ttr_2p5d['ttr'].isel(forecast_period=0) / 86400.0)
    # Day 2..N: use differences of the accumulated field
    olr_diffs = -ttr_2p5d['ttr'].diff('forecast_period') / 86400.0
    olr_2p5d = xr.concat([olr_day1, olr_diffs], dim='forecast_period')

    olr_2p5d = (
        olr_2p5d
        .assign_coords(time=(olr_2p5d['forecast_reference_time'] + olr_2p5d['forecast_period']))                              
        .swap_dims({'forecast_period': 'time'})                
        .drop_vars('forecast_period')                          
    )
    u200_2p5d = (
        u200_2p5d
        .assign_coords(time=(u200_2p5d['forecast_reference_time'] + u200_2p5d['forecast_period']))                              
        .swap_dims({'forecast_period': 'time'})                
        .drop_vars('forecast_period')                          
    )
    u850_2p5d = (
        u850_2p5d
        .assign_coords(time=(u850_2p5d['forecast_reference_time'] + u850_2p5d['forecast_period']))                              
        .swap_dims({'forecast_period': 'time'})                
        .drop_vars('forecast_period')                          
    )

    combined_olr_data_2p5d = xr.concat([gt_olr_data_2p5d_slice, olr_2p5d.to_dataset(name='olr')], dim='time')
    combined_u850_data_2p5d = xr.concat([gt_u850_data_2p5d_slice, u850_2p5d], dim='time')
    combined_u200_data_2p5d = xr.concat([gt_u200_data_2p5d_slice, u200_2p5d], dim='time')
    
    olr_seasonal_cycle = seasonal_cycle_ds['olr'].sel(dayofyear=combined_olr_data_2p5d.time.dt.dayofyear)
    u850_seasonal_cycle = seasonal_cycle_ds['u850'].sel(dayofyear=combined_u850_data_2p5d.time.dt.dayofyear)
    u200_seasonal_cycle = seasonal_cycle_ds['u200'].sel(dayofyear=combined_u200_data_2p5d.time.dt.dayofyear)

    # subtract mean and fisrt three harmonics
    olr_anomalies = combined_olr_data_2p5d - olr_seasonal_cycle
    u850_anomalies = combined_u850_data_2p5d - u850_seasonal_cycle
    u200_anomalies = combined_u200_data_2p5d - u200_seasonal_cycle

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
        filename=os.path.join(save_dir, f'{init_date}.txt'),
        method_str='ECMWF'
    )


if __name__ == "__main__":
    main()
