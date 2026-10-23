import os
import argparse
import warnings
import xarray as xr
import numpy as np
from mjo.utils.RMM.eof import load_olr, regrid_to_2p5, project_rmm
from mjo.utils.RMM.io import save_rmm_indices

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS.*")


def compute_ensemble_mean(forecast_root_dir):
    """Compute ensemble means of ECMWF S2S ttr and u forecasts and save them next to the inputs."""
    ds_ttr = xr.open_dataset(os.path.join(forecast_root_dir, 'ttr', 'ttr.nc'))
    ds_u = xr.open_dataset(os.path.join(forecast_root_dir, 'u', 'u.nc'))

    # Ensemble means over the 'number' dimension
    ttr_out = xr.Dataset({'ttr': ds_ttr['ttr'].mean('number', keep_attrs=True)})
    u_out = xr.Dataset({'u': ds_u['u'].mean('number', keep_attrs=True)})

    ttr_path = os.path.join(forecast_root_dir, 'ttr', 'ttr_ensmean.nc')
    u_path = os.path.join(forecast_root_dir, 'u', 'u_ensmean.nc')
    ttr_out.to_netcdf(ttr_path)
    u_out.to_netcdf(u_path)

    ds_ttr.close()
    ds_u.close()
    print(f"Saved:\n  {ttr_path}\n  {u_path}")


def to_absolute_time(ds):
    """Replace the forecast_period dimension with absolute valid time."""
    return (
        ds
        .assign_coords(time=(ds['forecast_reference_time'] + ds['forecast_period']))
        .swap_dims({'forecast_period': 'time'})
        .drop_vars('forecast_period')
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute RMM indices from an ECMWF S2S ensemble-mean forecast using precomputed EOFs.')
    parser.add_argument('--olr', required=True, help='NOAA interpolated daily OLR netCDF (olr.day.mean.nc), used for history')
    parser.add_argument('--era5', required=True, help='ERA5 daily zarr with u_component_of_wind, used for history')
    parser.add_argument('--eof-dir', required=True, help='Directory with eof.nc, seasonal_cycle.nc, normalization_factor.nc (from RMM/eof.py)')
    parser.add_argument('--forecast-dir', required=True, help='ECMWF S2S root containing ttr/ and u/ subdirectories')
    parser.add_argument('--output-dir', required=True, help='Directory to write <init-date>.txt')
    parser.add_argument('--init-date', default='2017-02-01', help='Forecast initialization date (default: %(default)s)')
    parser.add_argument('--compute-ensemble-mean', action='store_true', help='First compute ttr_ensmean.nc/u_ensmean.nc from ttr.nc/u.nc')
    return parser.parse_args()


def main():
    """Compute RMM indices from ECMWF S2S forecasts using precomputed EOFs."""
    args = parse_args()
    forecast_root_dir = args.forecast_dir
    init_date = args.init_date
    os.makedirs(args.output_dir, exist_ok=True)

    if args.compute_ensemble_mean:
        compute_ensemble_mean(forecast_root_dir)

    # Load reference datasets and EOFs
    raw_olr_ds = load_olr(args.olr)
    raw_era5_ds = xr.open_zarr(args.era5)
    seasonal_cycle_ds = xr.open_dataset(os.path.join(args.eof_dir, 'seasonal_cycle.nc')).load()
    normalization_factor_ds = xr.open_dataset(os.path.join(args.eof_dir, 'normalization_factor.nc')).load()
    EOF_ds = xr.open_dataset(os.path.join(args.eof_dir, 'eof.nc')).load()

    gt_olr_data = raw_olr_ds['olr'].to_dataset().load()
    gt_u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850').load()
    gt_u200_data = raw_era5_ds['u_component_of_wind'].sel(level=200, drop=True).to_dataset(name='u200').load()

    raw_olr_ds.close()
    raw_era5_ds.close()

    # Regrid observed history to 2.5°
    gt_olr_data_2p5d = regrid_to_2p5(gt_olr_data)
    gt_u850_data_2p5d = regrid_to_2p5(gt_u850_data)
    gt_u200_data_2p5d = regrid_to_2p5(gt_u200_data)

    # Load ensemble-mean forecast
    ds_ttr = xr.open_dataset(os.path.join(forecast_root_dir, 'ttr', 'ttr_ensmean.nc')).rename_dims(latitude='lat', longitude='lon')
    ds_u = xr.open_dataset(os.path.join(forecast_root_dir, 'u', 'u_ensmean.nc')).rename_dims(latitude='lat', longitude='lon')

    ttr = ds_ttr['ttr'].sel(forecast_reference_time=init_date).to_dataset(name='ttr')
    u200 = ds_u['u'].sel(pressure_level=200, forecast_reference_time=init_date).to_dataset(name='u200').drop_vars('pressure_level')
    u850 = ds_u['u'].sel(pressure_level=850, forecast_reference_time=init_date).to_dataset(name='u850').drop_vars('pressure_level')

    # Regrid forecast to 2.5°
    ttr_2p5d = regrid_to_2p5(ttr)
    u200_2p5d = regrid_to_2p5(u200)
    u850_2p5d = regrid_to_2p5(u850)

    # Observed history: 119 days up to and including the init date
    gt_end_date = np.datetime64(init_date)
    gt_start_date = gt_end_date - np.timedelta64(118, 'D')
    period = slice(gt_start_date, gt_end_date)
    gt_olr_data_2p5d_slice = gt_olr_data_2p5d.sel(time=period)
    gt_u850_data_2p5d_slice = gt_u850_data_2p5d.sel(time=period)
    gt_u200_data_2p5d_slice = gt_u200_data_2p5d.sel(time=period)

    # Convert accumulated top thermal radiation to daily OLR
    # Day 1: use first forecast step (convert from J/m² to W/m²)
    olr_day1 = (-ttr_2p5d['ttr'].isel(forecast_period=0) / 86400.0)
    # Day 2+: use differences of accumulated field
    olr_diffs = -ttr_2p5d['ttr'].diff('forecast_period') / 86400.0
    olr_2p5d = xr.concat([olr_day1, olr_diffs], dim='forecast_period')

    olr_2p5d = to_absolute_time(olr_2p5d)
    u200_2p5d = to_absolute_time(u200_2p5d)
    u850_2p5d = to_absolute_time(u850_2p5d)

    # Combine observed history with forecast
    combined_olr_data_2p5d = xr.concat([gt_olr_data_2p5d_slice, olr_2p5d.to_dataset(name='olr')], dim='time')
    combined_u850_data_2p5d = xr.concat([gt_u850_data_2p5d_slice, u850_2p5d], dim='time')
    combined_u200_data_2p5d = xr.concat([gt_u200_data_2p5d_slice, u200_2p5d], dim='time')

    time, RMM1, RMM2 = project_rmm(
        combined_olr_data_2p5d, combined_u850_data_2p5d, combined_u200_data_2p5d,
        seasonal_cycle_ds, normalization_factor_ds, EOF_ds
    )
    save_rmm_indices(
        time=time,
        RMM1=RMM1,
        RMM2=RMM2,
        filename=os.path.join(args.output_dir, f'{init_date}.txt'),
        method_str='ECMWF'
    )


if __name__ == "__main__":
    main()
