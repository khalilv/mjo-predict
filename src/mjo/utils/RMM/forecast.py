import os 
import glob
import xarray as xr
import numpy as np
import xesmf as xe
from mjo.utils.RMM.eof import detrend_anomalies, latitude_band_average
from mjo.utils.RMM.io import save_rmm_indices


def format(forecast_ds):
    init_time = forecast_ds['time'].isel(time=0)
    lead_days = xr.DataArray(forecast_ds['lead_time'].values.astype('timedelta64[D]'),
                            dims='lead_time')
    abs_time = init_time + lead_days
    fc_ds = (
        forecast_ds
        .isel(time=0, drop=True)  # drop singleton 'time'
        .assign_coords(time=abs_time)
        .swap_dims({'lead_time': 'time'})
        .drop_vars('lead_time')
    )
    channel_names = fc_ds['channel'].values
    var_data = {
        str(chan): fc_ds['__xarray_dataarray_variable__']
                    .sel(channel=chan)
                    .drop_vars('channel')  # each var becomes (time, lat, lon)
        for chan in channel_names
    }
    fc_ds = xr.Dataset(var_data)
    return fc_ds

def main():

    #reference period to use   
    reference_start = '1979-09-07'
    reference_end = '2001-12-31'
    
    olr_file_path = '/glade/derecho/scratch/kvirji/DATA/NOAA/OLR/PSL_interpolated/olr.day.mean.nc'
    era5_file_path = '/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr'
    reference_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/EOF/reference_period_{reference_start}_to_{reference_end}'
    save_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/RMM/FuXi/reference_period_{reference_start}_to_{reference_end}'

    member_dir = '/glade/derecho/scratch/kvirji/DATA/MJO/FuXi/20020103/2002/20020103/member/00'

    os.makedirs(save_dir, exist_ok=True)

    #open required datasets
    raw_olr_ds = xr.open_dataset(olr_file_path).isel(time=slice(249, None)) #remove some corrupted dates
    raw_era5_ds = xr.open_zarr(era5_file_path)
    seasonal_cycle_ds = xr.open_dataset(os.path.join(reference_dir, 'seasonal_cycle.nc'))
    normalization_factor_ds = xr.open_dataset(os.path.join(reference_dir, 'normalization_factor.nc'))
    EOF_ds = xr.open_dataset(os.path.join(reference_dir, 'eof.nc'))

    gt_olr_data = raw_olr_ds['olr'].to_dataset()
    gt_u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850')
    gt_u200_data = raw_era5_ds['u_component_of_wind'].sel(level=200, drop=True).to_dataset(name='u200')

    forecast_files = sorted(glob.glob(os.path.join(member_dir, "*.nc")))
    forecast_ds = xr.open_mfdataset(forecast_files, combine='by_coords', parallel=True)
    init_time = forecast_ds['time'].isel(time=0)

    #format forecast_ds to (time, lat, lon) for each variable
    forecast_ds = format(forecast_ds)
 
    #slide and load ground truth data for required period
    gt_end_date = init_time
    gt_start_date = gt_end_date - np.timedelta64(118, 'D')
    period = slice(gt_start_date, gt_end_date)
    gt_olr_data = gt_olr_data.sel(time=period).load()
    gt_u850_data = gt_u850_data.sel(time=period).load()
    gt_u200_data  = gt_u200_data .sel(time=period).load()

    # get forecast data for required variables
    forecast_olr_data = (forecast_ds['ttr'] * -1).to_dataset(name='olr')
    forecast_u850_data = forecast_ds['u850'].to_dataset(name='u850')
    forecast_u200_data = forecast_ds['u250'].to_dataset(name='u200') #TODO Update either the ground truth to use U250 or wait until FuXi releases U200

    # regrid to 2.5°
    target_lat = np.arange(-90, 90, 2.5)
    target_lon = np.arange(0, 360, 2.5)
    gt_olr_regridder = xe.Regridder(gt_olr_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    gt_u850_regridder = xe.Regridder(gt_u850_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    gt_u200_regridder = xe.Regridder(gt_u200_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    forecast_olr_regridder = xe.Regridder(forecast_olr_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    forecast_u850_regridder = xe.Regridder(forecast_u850_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    forecast_u200_regridder = xe.Regridder(forecast_u200_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)

    gt_olr_data_2p5d = gt_olr_regridder(gt_olr_data)
    gt_u850_data_2p5d = gt_u850_regridder(gt_u850_data)
    gt_u200_data_2p5d = gt_u200_regridder(gt_u200_data)

    forecast_olr_data_2p5d = forecast_olr_regridder(forecast_olr_data)
    forecast_u850_data_2p5d = forecast_u850_regridder(forecast_u850_data)
    forecast_u200_data_2p5d = forecast_u200_regridder(forecast_u200_data)

    combined_olr_data_2p5d = xr.concat([gt_olr_data_2p5d, forecast_olr_data_2p5d], dim='time')
    combined_u850_data_2p5d = xr.concat([gt_u850_data_2p5d, forecast_u850_data_2p5d], dim='time')
    combined_u200_data_2p5d = xr.concat([gt_u200_data_2p5d, forecast_u200_data_2p5d], dim='time')

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
        filename=os.path.join(save_dir, 'rmm.txt'),
        method_str='FuXi'
    )


if __name__ == "__main__":
    main()
