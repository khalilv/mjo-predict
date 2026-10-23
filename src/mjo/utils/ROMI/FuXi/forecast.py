import os
import glob
import argparse
import warnings
import xarray as xr
import numpy as np
import xesmf as xe
from tqdm import tqdm
from mjo.utils.RMM.eof import TARGET_LAT, TARGET_LON, load_olr, regrid_to_2p5
from mjo.utils.ROMI.eof import LAT_MIN, LAT_MAX, fill_and_drop, romi_smooth, project_romi, load_romi_eofs
from mjo.utils.ROMI.io import save_romi_indices
from mjo.utils.RMM.FuXi.utils import walk_to_forecast_dir

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS.*")

# Observed warm-up days: 40 (trailing mean) + 9 (running mean)
HISTORY_DAYS = 49


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute ROMI from FuXi-S2S ensemble-mean OLR forecasts using precomputed per-DOY EOFs.')
    parser.add_argument('--olr', required=True, help='NOAA interpolated daily OLR netCDF (olr.day.mean.nc), used for history')
    parser.add_argument('--eof-dir', required=True, help='Directory with eof.nc and seasonal_cycle.nc (from ROMI/eof.py)')
    parser.add_argument('--forecast-dir', required=True, help='Raw FuXi-S2S output root with <YYYYMMDD>/.../member/mean/ (from FuXi/mean.py)')
    parser.add_argument('--output-dir', required=True, help='Output root; writes <YYYY-MM-DD>/mean.txt')
    return parser.parse_args()


def main():
    """Compute ROMI indices from FuXi ensemble-mean OLR forecasts using precomputed per-DOY EOFs.

    Only the ensemble mean is processed; dates without a `mean/.done` sentinel are skipped.
    Preprocessing matches observation.py: seasonal cycle removal, 40-day trailing mean removal,
    9-day trailing running mean, projection onto per-DOY EOFs normalized by sigma.
    """
    args = parse_args()
    forecast_root_dir = args.forecast_dir
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # Load reference data
    seasonal_cycle_ds = xr.open_dataset(os.path.join(args.eof_dir, 'seasonal_cycle.nc')).load()
    eofs_flat, sigma = load_romi_eofs(args.eof_dir)

    # Load observed OLR for history, regrid to 2.5 deg and subset to 20S-20N
    raw_olr_ds = load_olr(args.olr)
    gt_olr_data = raw_olr_ds['olr'].to_dataset().load()
    raw_olr_ds.close()
    gt_olr_2p5 = regrid_to_2p5(gt_olr_data).sel(lat=slice(LAT_MIN, LAT_MAX))

    start_dates = sorted(
        d for d in os.listdir(forecast_root_dir)
        if os.path.isdir(os.path.join(forecast_root_dir, d)) and d.isdigit() and len(d) == 8
    )
    forecast_regridder = None

    for start_date in tqdm(start_dates, 'Processing init dates'):
        root = os.path.join(forecast_root_dir, start_date)
        forecast_dir = walk_to_forecast_dir(root)
        mean_dir = os.path.join(forecast_dir, 'mean')

        # Skip dates whose mean isn't finished yet
        if not os.path.exists(os.path.join(mean_dir, '.done')):
            print(f'skip {start_date}: mean not ready', flush=True)
            continue

        out_dir_name = f'{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}'
        out_dir = os.path.join(save_dir, out_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'mean.txt')
        if os.path.exists(out_path):
            continue

        # Load ensemble-mean forecast (already formatted by mean.py: absolute time, channels split)
        forecast_files = sorted(glob.glob(os.path.join(mean_dir, '*.nc')))
        forecast_ds = xr.open_mfdataset(forecast_files, combine='by_coords', parallel=False).load()
        init_time = np.datetime64(out_dir_name)

        # Regrid forecast OLR to 2.5 deg (weights reused across dates), subset to 20S-20N
        if not forecast_regridder:
            forecast_regridder = xe.Regridder(forecast_ds, {'lat': TARGET_LAT, 'lon': TARGET_LON}, 'bilinear', periodic=True)
        forecast_2p5 = forecast_regridder(forecast_ds)
        forecast_olr = (forecast_2p5['ttr'] * -1).to_dataset(name='olr').sel(lat=slice(LAT_MIN, LAT_MAX))
        forecast_ds.close()

        # Concatenate observed history (up to and including init date) with forecast
        gt_start = init_time - np.timedelta64(HISTORY_DAYS - 1, 'D')
        gt_slice = gt_olr_2p5.sel(time=slice(gt_start, init_time))
        combined = xr.concat([gt_slice, forecast_olr], dim='time')

        # Remove seasonal cycle
        seasonal = seasonal_cycle_ds['olr'].sel(dayofyear=combined.time.dt.dayofyear)
        anom = (combined - seasonal)['olr']

        # Fill missing days, filter, and keep only forecast days (after init date)
        anom_np, times, doys = fill_and_drop(anom)
        smoothed = romi_smooth(anom_np.reshape(anom_np.shape[0], -1))
        forecast_mask = times > init_time

        romi1, romi2 = project_romi(smoothed[forecast_mask], doys[forecast_mask], eofs_flat, sigma)
        save_romi_indices(
            time=times[forecast_mask], ROMI1=romi1, ROMI2=romi2,
            filename=out_path, method_str='FuXi_ROMI_mean',
        )


if __name__ == "__main__":
    main()
