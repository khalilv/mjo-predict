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


def main():
    """Compute RMM indices from FuXi forecasts using precomputed EOFs."""
    olr_file_path = '/glade/derecho/scratch/kvirji/DATA/NOAA/OLR/PSL_interpolated/olr.day.mean.nc'
    era5_file_path = '/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr'
    reference_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/U200/EOF'
    save_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/U200/FuXi'
    forecast_root_dir = '/glade/derecho/scratch/kvirji/DATA/MJO/FuXi/'

    os.makedirs(save_dir, exist_ok=True)

    # Load reference datasets and EOFs
    raw_olr_ds = xr.open_dataset(olr_file_path).isel(time=slice(249, None))  # Remove corrupted dates
    raw_era5_ds = xr.open_zarr(era5_file_path)
    seasonal_cycle_ds = xr.open_dataset(os.path.join(reference_dir, 'seasonal_cycle.nc')).load()
    normalization_factor_ds = xr.open_dataset(os.path.join(reference_dir, 'normalization_factor.nc')).load()
    EOF_ds = xr.open_dataset(os.path.join(reference_dir, 'eof.nc')).load()

    gt_olr_data = raw_olr_ds['olr'].to_dataset().load()
    gt_u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850').load()
    gt_u200_data = raw_era5_ds['u_component_of_wind'].sel(level=200, drop=True).to_dataset(name='u200').load()

    raw_olr_ds.close()
    raw_era5_ds.close()

    # Regrid ground truth data to 2.5°
    target_lat = np.arange(-90, 90, 2.5)
    target_lon = np.arange(0, 360, 2.5)
    gt_olr_regridder = xe.Regridder(gt_olr_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    gt_u850_regridder = xe.Regridder(gt_u850_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    gt_u200_regridder = xe.Regridder(gt_u200_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    gt_olr_data_2p5d = gt_olr_regridder(gt_olr_data)
    gt_u850_data_2p5d = gt_u850_regridder(gt_u850_data)
    gt_u200_data_2p5d = gt_u200_regridder(gt_u200_data)

    start_dates = sorted(os.listdir(forecast_root_dir))

    # Process each forecast start date and ensemble member
    forecast_regridder = None
    for start_date in start_dates:
        root = os.path.join(forecast_root_dir, start_date)
        if os.path.isdir(root) and not start_date.startswith('.'):
            # Navigate to member directory
            forecast_dir = walk_to_forecast_dir(root)

            for member in tqdm(range(51), f'Processing ensemble members for {forecast_dir}'):
                member_str = f"{member:02d}"

                # Load forecast data
                forecast_files = sorted(glob.glob(os.path.join(forecast_dir, member_str, "*.nc")))
                forecast_ds = xr.open_mfdataset(forecast_files, combine='by_coords', parallel=False)
                init_time = forecast_ds['time'].isel(time=0)

                # Create output directory for this initialization date
                member_dir = os.path.join(save_dir, f'{str(init_time.time.values).split("T")[0]}')
                os.makedirs(member_dir, exist_ok=True)

                if os.path.exists(os.path.join(member_dir, f'{member_str}.txt')):
                    print(os.path.join(member_dir, f'{member_str}.txt') + ' already exists.')
                    continue

                # Format forecast to (time, lat, lon) for each variable
                forecast_ds = format(forecast_ds.load())
                forecast_ds.close()

                # Extract ground truth data for 119 days before forecast init
                gt_end_date = init_time
                gt_start_date = gt_end_date - np.timedelta64(118, 'D')
                period = slice(gt_start_date, gt_end_date)
                gt_olr_data_2p5d_slice = gt_olr_data_2p5d.sel(time=period)
                gt_u850_data_2p5d_slice = gt_u850_data_2p5d.sel(time=period)
                gt_u200_data_2p5d_slice  = gt_u200_data_2p5d.sel(time=period)

                # Regrid forecast to 2.5°
                if not forecast_regridder:
                    forecast_regridder = xe.Regridder(forecast_ds, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
                forecast_ds_2p5d = forecast_regridder(forecast_ds)

                # Extract required variables (convert TTR to OLR by negation)
                forecast_olr_data_2p5d = (forecast_ds_2p5d['ttr'] * -1).to_dataset(name='olr')
                forecast_u850_data_2p5d = forecast_ds_2p5d['u850'].to_dataset(name='u850')
                forecast_u200_data_2p5d = forecast_ds_2p5d['u250'].to_dataset(name='u200')

                # Combine ground truth history with forecast
                combined_olr_data_2p5d = xr.concat([gt_olr_data_2p5d_slice, forecast_olr_data_2p5d], dim='time')
                combined_u850_data_2p5d = xr.concat([gt_u850_data_2p5d_slice, forecast_u850_data_2p5d], dim='time')
                combined_u200_data_2p5d = xr.concat([gt_u200_data_2p5d_slice, forecast_u200_data_2p5d], dim='time')

                # Remove seasonal cycle
                olr_seasonal_cycle = seasonal_cycle_ds['olr'].sel(dayofyear=combined_olr_data_2p5d.time.dt.dayofyear)
                u850_seasonal_cycle = seasonal_cycle_ds['u850'].sel(dayofyear=combined_u850_data_2p5d.time.dt.dayofyear)
                u200_seasonal_cycle = seasonal_cycle_ds['u200'].sel(dayofyear=combined_u200_data_2p5d.time.dt.dayofyear)

                olr_anomalies = combined_olr_data_2p5d - olr_seasonal_cycle
                u850_anomalies = combined_u850_data_2p5d - u850_seasonal_cycle
                u200_anomalies = combined_u200_data_2p5d - u200_seasonal_cycle

                # Detrend using 120-day running mean
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
                    filename=os.path.join(member_dir, f'{member_str}.txt'),
                    method_str='FuXi'
                )

if __name__ == "__main__":
    main()
