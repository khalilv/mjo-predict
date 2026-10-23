import os
import glob
import argparse
import xarray as xr
from tqdm import tqdm
from mjo.utils.RMM.FuXi.utils import walk_to_forecast_dir, format_forecast


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute the FuXi-S2S ensemble mean for each init date and save it under <member dir>/mean/.')
    parser.add_argument('--forecast-dir', required=True, help='Raw FuXi-S2S output root with one subdirectory per init date (from FuXi/generate.py)')
    parser.add_argument('--members', type=int, default=51, help='Number of ensemble members (default: %(default)s)')
    return parser.parse_args()


def main():
    """Compute ensemble mean from FuXi forecast members and save by lead time."""
    args = parse_args()
    root_dir = args.forecast_dir
    start_dates = sorted(os.listdir(root_dir))

    for start_date in start_dates:
        root = os.path.join(root_dir, start_date)
        if not os.path.isdir(root) or start_date.startswith('.'):
            continue
        ensemble_dir = walk_to_forecast_dir(root)

        # Load all ensemble members
        member_datasets = []
        for member in tqdm(range(args.members), f'Loading data for {start_date}'):
            member_str = f"{member:02d}"
            forecast_files = sorted(glob.glob(os.path.join(ensemble_dir, member_str, "*.nc")))
            member_ds = xr.open_mfdataset(forecast_files, combine='by_coords', parallel=False)
            member_ds = member_ds.expand_dims({"member": [int(member)]})
            member_datasets.append(member_ds)

        # Concatenate and format all members
        forecast_ds = xr.concat(member_datasets, dim='member')
        forecast_ds = format_forecast(forecast_ds)

        for ds in member_datasets:
            ds.close()

        ensemble_mean = forecast_ds.mean(dim='member')

        mean_dir = os.path.join(ensemble_dir, "mean")
        os.makedirs(mean_dir, exist_ok=True)

        # Save one file per time step (lead time)
        for i in tqdm(range(ensemble_mean.dims["time"]), "Saving mean forecast"):
            timestep_ds = ensemble_mean.isel(time=i).expand_dims("time")
            timestep_filename = os.path.join(mean_dir, f"{i+1:02d}.nc")
            timestep_ds.to_netcdf(timestep_filename)


if __name__ == "__main__":
    main()
