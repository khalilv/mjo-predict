import os
import xarray as xr
import numpy as np
import xesmf as xe
from mjo.utils.RMM.eof import detrend_anomalies, latitude_band_average
from mjo.utils.RMM.io import save_rmm_indices


def main():
    """Compute RMM indices from observed OLR and ERA5 wind data using precomputed EOFs."""
    # Reference period used for computing EOFs
    reference_start = '1979-09-07'
    reference_end = '2001-12-31'

    # Date range for computing RMM indices
    start_date = '1979-09-07'
    end_date = '2022-12-31'

    # File paths
    olr_file_path = '/glade/derecho/scratch/kvirji/DATA/NOAA/OLR/PSL_interpolated/olr.day.mean.nc'
    era5_file_path = '/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr'
    reference_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/U250/EOF'
    save_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM'

    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    raw_olr_ds = xr.open_dataset(olr_file_path).isel(time=slice(249, None))  # Remove corrupted dates
    raw_era5_ds = xr.open_zarr(era5_file_path)
    seasonal_cycle_ds = xr.open_dataset(os.path.join(reference_dir, 'seasonal_cycle.nc'))
    normalization_factor_ds = xr.open_dataset(os.path.join(reference_dir, 'normalization_factor.nc'))
    EOF_ds = xr.open_dataset(os.path.join(reference_dir, 'eof.nc'))

    # Extract wind components
    olr_data = raw_olr_ds['olr'].to_dataset()
    u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850')
    u200_data = raw_era5_ds['u_component_of_wind'].sel(level=200, drop=True).to_dataset(name='u200')

    # Regrid to 2.5° resolution to match EOF calculation
    target_lat = np.arange(-90, 90, 2.5)
    target_lon = np.arange(0, 360, 2.5)
    olr_regridder = xe.Regridder(olr_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    u850_regridder = xe.Regridder(u850_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    u200_regridder = xe.Regridder(u200_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)

    olr_data_2p5d = olr_regridder(olr_data)
    u850_data_2p5d = u850_regridder(u850_data)
    u200_data_2p5d = u200_regridder(u200_data)

    # Slice and load data for required period
    period = slice(start_date, end_date)
    olr_data_2p5d = olr_data_2p5d.sel(time=period).load()
    u850_data_2p5d = u850_data_2p5d.sel(time=period).load()
    u200_data_2p5d = u200_data_2p5d.sel(time=period).load()

    # Remove seasonal cycle
    olr_seasonal_cycle = seasonal_cycle_ds['olr'].sel(dayofyear=olr_data_2p5d.time.dt.dayofyear)
    u850_seasonal_cycle = seasonal_cycle_ds['u850'].sel(dayofyear=u850_data_2p5d.time.dt.dayofyear)
    u200_seasonal_cycle = seasonal_cycle_ds['u200'].sel(dayofyear=u200_data_2p5d.time.dt.dayofyear)

    olr_anomalies = olr_data_2p5d - olr_seasonal_cycle
    u850_anomalies = u850_data_2p5d - u850_seasonal_cycle
    u200_anomalies = u200_data_2p5d - u200_seasonal_cycle

    # Detrend anomalies using 120-day running mean
    detrended_olr_anomalies = detrend_anomalies(olr_anomalies)
    detrended_u850_anomalies = detrend_anomalies(u850_anomalies)
    detrended_u200_anomalies = detrend_anomalies(u200_anomalies)

    # Average over 15°S-15°N latitude band
    detrended_olr_anomalies_latitude_band_avg = latitude_band_average(detrended_olr_anomalies)
    detrended_u850_anomalies_latitude_band_avg = latitude_band_average(detrended_u850_anomalies)
    detrended_u200_anomalies_latitude_band_avg = latitude_band_average(detrended_u200_anomalies)

    # Normalize using factors from reference period
    detrended_olr_anomalies_latitude_band_avg_norm = detrended_olr_anomalies_latitude_band_avg / normalization_factor_ds['olr']
    detrended_u850_anomalies_latitude_band_avg_norm = detrended_u850_anomalies_latitude_band_avg / normalization_factor_ds['u850']
    detrended_u200_anomalies_latitude_band_avg_norm = detrended_u200_anomalies_latitude_band_avg / normalization_factor_ds['u200']

    # Drop missing values
    detrended_olr_anomalies_latitude_band_avg_norm = detrended_olr_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')
    detrended_u850_anomalies_latitude_band_avg_norm = detrended_u850_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')
    detrended_u200_anomalies_latitude_band_avg_norm = detrended_u200_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')

    # Align timesteps across all variables
    olr, u850, u200 = xr.align(
        detrended_olr_anomalies_latitude_band_avg_norm,
        detrended_u850_anomalies_latitude_band_avg_norm,
        detrended_u200_anomalies_latitude_band_avg_norm,
        join='inner'
    )

    # Combine data into single array (time, 3 × lon)
    X = xr.concat([olr['olr'], u850['u850'], u200['u200']], dim='lon')

    # Project onto reference EOFs to compute RMM indices
    RMM1 = X.values @ EOF_ds['EOF1'].values
    RMM2 = X.values @ EOF_ds['EOF2'].values

    # Normalize RMM indices using reference period standard deviations
    RMM1_norm = RMM1 / normalization_factor_ds['RMM1_std'].values
    RMM2_norm = RMM2 / normalization_factor_ds['RMM2_std'].values

    # Save indices to text file
    save_rmm_indices(
        time=olr.time.values,
        RMM1=RMM1_norm,
        RMM2=RMM2_norm,
        filename=os.path.join(save_dir, 'rmm.txt')
    )


if __name__ == "__main__":
    main()
