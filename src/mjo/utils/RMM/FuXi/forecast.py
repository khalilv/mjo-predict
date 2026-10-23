import os
import glob
import argparse
import warnings
import xarray as xr
import numpy as np
import xesmf as xe
from tqdm import tqdm
from mjo.utils.RMM.eof import TARGET_LAT, TARGET_LON, load_olr, regrid_to_2p5, project_rmm
from mjo.utils.RMM.io import save_rmm_indices
from mjo.utils.RMM.FuXi.utils import format_forecast, walk_to_forecast_dir

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS.*")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute RMM indices for each FuXi-S2S ensemble member using precomputed EOFs.')
    parser.add_argument('--olr', required=True, help='NOAA interpolated daily OLR netCDF (olr.day.mean.nc), used for history')
    parser.add_argument('--era5', required=True, help='ERA5 daily zarr with u_component_of_wind, used for history')
    parser.add_argument('--eof-dir', required=True, help='Directory with eof.nc, seasonal_cycle.nc, normalization_factor.nc (from RMM/eof.py)')
    parser.add_argument('--forecast-dir', required=True, help='Raw FuXi-S2S output root with one subdirectory per init date (from FuXi/generate.py)')
    parser.add_argument('--upper-level', type=int, default=250, help='Upper-tropospheric wind level in hPa; FuXi-S2S provides u250 (default: %(default)s)')
    parser.add_argument('--output-dir', default='data/FuXi', help='Output root; writes <YYYY-MM-DD>/<member>.txt (default: %(default)s)')
    parser.add_argument('--members', type=int, default=51, help='Number of ensemble members (default: %(default)s)')
    return parser.parse_args()


def main():
    """Compute RMM indices from FuXi forecasts using precomputed EOFs."""
    args = parse_args()
    forecast_root_dir = args.forecast_dir
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # Load reference datasets and EOFs
    raw_olr_ds = load_olr(args.olr)
    raw_era5_ds = xr.open_zarr(args.era5)
    seasonal_cycle_ds = xr.open_dataset(os.path.join(args.eof_dir, 'seasonal_cycle.nc')).load()
    normalization_factor_ds = xr.open_dataset(os.path.join(args.eof_dir, 'normalization_factor.nc')).load()
    EOF_ds = xr.open_dataset(os.path.join(args.eof_dir, 'eof.nc')).load()

    gt_olr_data = raw_olr_ds['olr'].to_dataset().load()
    gt_u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850').load()
    gt_u_upper_data = raw_era5_ds['u_component_of_wind'].sel(level=args.upper_level, drop=True).to_dataset(name='u_upper').load()

    raw_olr_ds.close()
    raw_era5_ds.close()

    # Regrid observed history to 2.5°
    gt_olr_data_2p5d = regrid_to_2p5(gt_olr_data)
    gt_u850_data_2p5d = regrid_to_2p5(gt_u850_data)
    gt_u_upper_data_2p5d = regrid_to_2p5(gt_u_upper_data)

    start_dates = sorted(os.listdir(forecast_root_dir))

    # Process each forecast start date and ensemble member
    forecast_regridder = None
    for start_date in start_dates:
        root = os.path.join(forecast_root_dir, start_date)
        if not os.path.isdir(root) or start_date.startswith('.'):
            continue
        forecast_dir = walk_to_forecast_dir(root)

        for member in tqdm(range(args.members), f'Processing ensemble members for {forecast_dir}'):
            member_str = f"{member:02d}"

            # Load forecast data
            forecast_files = sorted(glob.glob(os.path.join(forecast_dir, member_str, "*.nc")))
            forecast_ds = xr.open_mfdataset(forecast_files, combine='by_coords', parallel=False)
            init_time = forecast_ds['time'].isel(time=0)

            # Create output directory for this initialization date
            member_dir = os.path.join(save_dir, f'{str(init_time.time.values).split("T")[0]}')
            os.makedirs(member_dir, exist_ok=True)

            out_path = os.path.join(member_dir, f'{member_str}.txt')
            if os.path.exists(out_path):
                print(out_path + ' already exists.')
                continue

            # Format forecast to (time, lat, lon) for each variable
            forecast_ds = format_forecast(forecast_ds.load())
            forecast_ds.close()

            # Observed history: 119 days up to and including the init date
            gt_end_date = init_time
            gt_start_date = gt_end_date - np.timedelta64(118, 'D')
            period = slice(gt_start_date, gt_end_date)
            gt_olr_data_2p5d_slice = gt_olr_data_2p5d.sel(time=period)
            gt_u850_data_2p5d_slice = gt_u850_data_2p5d.sel(time=period)
            gt_u_upper_data_2p5d_slice = gt_u_upper_data_2p5d.sel(time=period)

            # Regrid forecast to 2.5° (regridder weights are reused across members/dates)
            if not forecast_regridder:
                forecast_regridder = xe.Regridder(forecast_ds, {'lat': TARGET_LAT, 'lon': TARGET_LON}, 'bilinear', periodic=True)
            forecast_ds_2p5d = forecast_regridder(forecast_ds)

            # Extract required variables (convert TTR to OLR by negation).
            forecast_olr_data_2p5d = (forecast_ds_2p5d['ttr'] * -1).to_dataset(name='olr')
            forecast_u850_data_2p5d = forecast_ds_2p5d['u850'].to_dataset(name='u850')
            forecast_u_upper_data_2p5d = forecast_ds_2p5d[f'u{args.upper_level}'].to_dataset(name='u_upper')

            # Combine observed history with forecast
            combined_olr_data_2p5d = xr.concat([gt_olr_data_2p5d_slice, forecast_olr_data_2p5d], dim='time')
            combined_u850_data_2p5d = xr.concat([gt_u850_data_2p5d_slice, forecast_u850_data_2p5d], dim='time')
            combined_u_upper_data_2p5d = xr.concat([gt_u_upper_data_2p5d_slice, forecast_u_upper_data_2p5d], dim='time')

            time, RMM1, RMM2 = project_rmm(
                combined_olr_data_2p5d, combined_u850_data_2p5d, combined_u_upper_data_2p5d,
                seasonal_cycle_ds, normalization_factor_ds, EOF_ds
            )
            save_rmm_indices(time=time, RMM1=RMM1, RMM2=RMM2, filename=out_path, method_str='FuXi')


if __name__ == "__main__":
    main()
