import os
import argparse
import xarray as xr
from mjo.utils.RMM.eof import load_olr, regrid_to_2p5, project_rmm
from mjo.utils.RMM.io import save_rmm_indices


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute observed RMM indices from NOAA OLR and ERA5 winds using precomputed EOFs.')
    parser.add_argument('--olr', required=True, help='NOAA interpolated daily OLR netCDF (olr.day.mean.nc)')
    parser.add_argument('--era5', required=True, help='ERA5 daily zarr with u_component_of_wind on pressure levels')
    parser.add_argument('--eof-dir', required=True, help='Directory with eof.nc, seasonal_cycle.nc, normalization_factor.nc (from RMM/eof.py)')
    parser.add_argument('--upper-level', type=int, default=250, help='ERA5 upper-tropospheric wind level in hPa; must match the EOFs (default: %(default)s)')
    parser.add_argument('--output', default='data/observed/rmm.txt', help='Output RMM text file (default: %(default)s)')
    parser.add_argument('--start-date', default='1979-09-07', help='First date to compute (default: %(default)s)')
    parser.add_argument('--end-date', default='2022-12-31', help='Last date to compute (default: %(default)s)')
    return parser.parse_args()


def main():
    """Compute RMM indices from observed OLR and ERA5 wind data using precomputed EOFs."""
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Load datasets
    raw_olr_ds = load_olr(args.olr)
    raw_era5_ds = xr.open_zarr(args.era5)
    seasonal_cycle_ds = xr.open_dataset(os.path.join(args.eof_dir, 'seasonal_cycle.nc'))
    normalization_factor_ds = xr.open_dataset(os.path.join(args.eof_dir, 'normalization_factor.nc'))
    EOF_ds = xr.open_dataset(os.path.join(args.eof_dir, 'eof.nc'))

    # Extract variables
    olr_data = raw_olr_ds['olr'].to_dataset()
    u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850')
    u_upper_data = raw_era5_ds['u_component_of_wind'].sel(level=args.upper_level, drop=True).to_dataset(name='u_upper')

    # Regrid to 2.5° resolution and load the required period
    period = slice(args.start_date, args.end_date)
    olr_data_2p5d = regrid_to_2p5(olr_data).sel(time=period).load()
    u850_data_2p5d = regrid_to_2p5(u850_data).sel(time=period).load()
    u_upper_data_2p5d = regrid_to_2p5(u_upper_data).sel(time=period).load()

    time, RMM1, RMM2 = project_rmm(
        olr_data_2p5d, u850_data_2p5d, u_upper_data_2p5d,
        seasonal_cycle_ds, normalization_factor_ds, EOF_ds
    )
    save_rmm_indices(time=time, RMM1=RMM1, RMM2=RMM2, filename=args.output)


if __name__ == "__main__":
    main()
