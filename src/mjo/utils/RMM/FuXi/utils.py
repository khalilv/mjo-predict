import os
import xarray as xr 
import os

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

def walk_to_forecast_dir(root):
    current_dir = root
    while True:
        subdirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
        if 'member' in subdirs:
            forecast_dir = os.path.join(current_dir, 'member')
            break
        elif len(subdirs) == 1:
            current_dir = os.path.join(current_dir, subdirs[0])
        else:
            raise RuntimeError(f"Unexpected directory structure in {current_dir}: {subdirs}")
    return forecast_dir
