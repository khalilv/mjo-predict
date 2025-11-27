import os
import xarray as xr


def format(forecast_ds):
    """Convert FuXi forecast from lead_time to absolute time coordinates and split channels into variables."""
    # Convert lead time to absolute time
    init_time = forecast_ds['time'].isel(time=0)
    lead_days = xr.DataArray(forecast_ds['lead_time'].values.astype('timedelta64[D]'),
                            dims='lead_time')
    abs_time = init_time + lead_days

    # Swap lead_time dimension with absolute time
    fc_ds = (
        forecast_ds
        .isel(time=0, drop=True)
        .assign_coords(time=abs_time)
        .swap_dims({'lead_time': 'time'})
        .drop_vars('lead_time')
    )

    # Split channel dimension into separate variables
    channel_names = fc_ds['channel'].values
    var_data = {
        str(chan): fc_ds['__xarray_dataarray_variable__']
                    .sel(channel=chan)
                    .drop_vars('channel')
        for chan in channel_names
    }
    fc_ds = xr.Dataset(var_data)
    return fc_ds


def walk_to_forecast_dir(root):
    """Navigate directory tree to find the 'member' forecast directory."""
    current_dir = root
    while True:
        subdirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]

        # Found the member directory
        if 'member' in subdirs:
            forecast_dir = os.path.join(current_dir, 'member')
            break
        # Keep walking down if only one subdirectory exists
        elif len(subdirs) == 1:
            current_dir = os.path.join(current_dir, subdirs[0])
        else:
            raise RuntimeError(f"Unexpected directory structure in {current_dir}: {subdirs}")

    return forecast_dir
