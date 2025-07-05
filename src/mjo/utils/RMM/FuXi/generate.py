import os
import time 
import glob
import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe
import onnxruntime as ort
from copy import deepcopy

FUXI_VARIABLES = [
    ('geopotential', 'z'),
    ('temperature', 't'),
    ('u_component_of_wind', 'u'),
    ('v_component_of_wind', 'v'),
    ('specific_humidity', 'q'),
    ('2m_temperature', 't2m'),
    ('2m_dewpoint_temperature', 'd2m'),
    ('sea_surface_temperature', 'sst'),
    ('top_net_thermal_radiation','ttr'),
    ('10m_u_component_of_wind', '10u'),
    ('10m_v_component_of_wind', '10v'),
    ('100m_u_component_of_wind', '100u'),
    ('100m_v_component_of_wind', '100v'),
    ('mean_sea_level_pressure', 'msl'),
    ('total_column_water_vapour', 'tcwv'),
    ('total_precipitation_24hr', 'tp'),
]

def load_wb2_data(path):
    ds = xr.open_zarr(path)
    available_vars = [long for long,_ in FUXI_VARIABLES if long in ds.variables]
    ds = ds[available_vars]
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})
    return ds

def load_cds_data(path):
    ds = xr.open_mfdataset(glob.glob(os.path.join(path, "*.nc")), combine='by_coords', engine='netcdf4')
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})

    return ds

def load_olr_data(path):
    ds = xr.open_dataset(path).isel(time=slice(249, None)) # remove some corrupted dates   
    return ds

def regrid_to(ds, lat, lon):
    olr_regridder = xe.Regridder(ds, {'lat': lat, 'lon': lon}, 'bilinear', periodic=True)
    ds = olr_regridder(ds)
    return ds

def slice_to(ds, start_date, end_date):
    period = slice(start_date, end_date)
    ds = ds.sel(time=period)
    return ds

def format(ds):
    formatted_ds = []
    channel = []
    for (long, short) in FUXI_VARIABLES:
        v = ds[long]
        if 'level' not in v.dims:
            v = v.expand_dims({'level': [1]})

        if short == "tp":
            v = np.clip(v * 1000, 0, 1000)

        # elif short == "ttr":
        #     v = v / 3600

        if v.level.values[0] != 1000:
            v = v.reindex(level=v.level[::-1])

        if short in ['z', 't', 'u', 'v', 'q']:
            level = [f'{short}{l}' for l in v.level.values]
        else:
            level = [short]

        v.name = "data"
        v.attrs = {}        
        v = v.assign_coords(level=level)
        formatted_ds.append(v)
        channel += level

    formatted_ds = xr.concat(formatted_ds, 'level').rename({"level": "channel"})
    formatted_ds = formatted_ds.assign_coords(channel=channel)
    return formatted_ds

def clean(ds):
    null_times = None
    for var in ds.data_vars:
        if var == 'sea_surface_temperature': continue
        arr = ds[var]
        if 'time' in arr.dims:
            other_dims = [d for d in arr.dims if d != 'time']
            if other_dims:
                mask = arr.isnull().any(dim=other_dims)
            else:
                mask = arr.isnull()
            times_with_null = arr['time'].where(mask, drop=True)
            if null_times is None:
                null_times = times_with_null
            else:
                null_times = xr.concat([null_times, times_with_null], dim='time')
    if null_times is not None and null_times.size > 0:
        prev_times = null_times - np.timedelta64(1, 'D')
        to_remove = xr.concat([null_times, prev_times], dim='time').drop_duplicates('time')
        ds = ds.sel(time=~ds['time'].isin(to_remove))
    return ds

def print_dataarray(ds, msg='', n=10):
    tid = np.arange(0, ds.shape[0])
    tid = np.append(tid[:n], tid[-n:])    
    v = ds.isel(time=tid)
    msg += f"short_name: {ds.name}, shape: {ds.shape}, value: {v.values.min():.3f} ~ {v.values.max():.3f}"
    
    if 'lat' in ds.dims:
        lat = ds.lat.values
        msg += f", lat: {lat[0]:.3f} ~ {lat[-1]:.3f}"
    if 'lon' in ds.dims:
        lon = ds.lon.values
        msg += f", lon: {lon[0]:.3f} ~ {lon[-1]:.3f}"   

    if "level" in v.dims and len(v.level) > 1:
        for lvl in v.level.values:
            x = v.sel(level=lvl).values
            msg += f"\nlevel: {lvl:04d}, value: {x.min():.3f} ~ {x.max():.3f}"

    if "channel" in v.dims and len(v.channel) > 1:
        for ch in v.channel.values:
            x = v.sel(channel=ch).values
            msg += f"\nchannel: {ch}, value: {x.min():.3f} ~ {x.max():.3f}"

    print(msg)

def save_with_progress(ds, save_name, dtype=np.float32):
    from dask.diagnostics import ProgressBar

    if 'time' in ds.dims:
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))

    ds = ds.astype(dtype)
    obj = ds.to_netcdf(save_name, compute=False)

    with ProgressBar():
        obj.compute()


def save_like(output, input, member, lead_time, save_dir=""):

    if save_dir:
        save_dir = os.path.join(save_dir, f"member/{member:02d}")
        os.makedirs(save_dir, exist_ok=True)
        init_time = pd.to_datetime(input.time.data[-1])

        ds = xr.DataArray(
            data=output,
            dims=['time', 'lead_time', 'channel', 'lat', 'lon'],
            coords=dict(
                time=[init_time],
                lead_time=[lead_time],
                channel=input.channel,
                lat=input.lat,
                lon=input.lon,
            )
        ).astype(np.float32)
        ds = ds.sel(channel=['u250', 'u200', 'u850', 'ttr'])
        # print_dataarray(ds)
        save_name = os.path.join(save_dir, f'{lead_time:02d}.nc')
        ds.to_netcdf(save_name)



def load_model(model_name, device):
    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    
    if device == "cuda":
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy':'kSameAsRequested'})]
    elif device == "cpu":
        providers=['CPUExecutionProvider']
        options.intra_op_num_threads = 24
    else:
        raise ValueError("device must be cpu or cuda!")

    session = ort.InferenceSession(
        model_name,  
        sess_options=options, 
        providers=providers
    )
    return session


def run_inference(
    model, 
    input, 
    total_step, 
    total_member, 
    time_strs,
    save_dir=""
):
    input_names = [input.name for input in model.get_inputs()]
    # hist_time = pd.to_datetime(input.time.values[-2])
    # init_time = pd.to_datetime(input.time.values[-1])
    # assert init_time - hist_time == pd.Timedelta(days=1)
    
    lat = input.lat.values 
    lon = input.lon.values 
    batch = input.values
    if batch.shape[3] > batch.shape[4]:
        batch = batch.swapaxes(3,4) #(B, T, V, H, W)
    
    assert lat[0] == 90 and lat[-1] == -90
    print(f"Region: {lat[0]:.2f} ~ {lat[-1]:.2f}, {lon[0]:.2f} ~ {lon[-1]:.2f}")

    for member in range(total_member):
        print(f'Inference member {member:02d} ...')
        new_input = deepcopy(batch)

        start = time.perf_counter()
        for step in range(total_step):
            lead_time = (step + 1)

            inputs = {'input': new_input}        

            if "step" in input_names:
                inputs['step'] = np.array([step], dtype=np.float32)

            # if "doy" in input_names:
            #     valid_time = init_time + pd.Timedelta(days=step)
            #     doy = min(365, valid_time.day_of_year)/365 
            #     inputs['doy'] = np.array([doy], dtype=np.float32)

            istart = time.perf_counter()
            new_input, = model.run(None, inputs)
            output = deepcopy(new_input[:, -1:])
            step_time = time.perf_counter() - istart

            print(f"member: {member:02d}, step {step+1:02d}, step_time: {step_time:.3f} sec")
            for b in range(output.shape[0]):
                save = os.path.join(save_dir, time_strs[b])
                os.makedirs(save, exist_ok=True)
                save_like(output[b:b+1], input, member, lead_time, save)
            
            if step > total_step:
                break

        run_time = time.perf_counter() - start
        print(f'Inference member done, take {run_time:.2f} sec')

def batch_input(input, batch_size):
    batch = []
    init_time_strs = []
    time_values = input.time.values

    for i in range(1, len(time_values)):
        t0 = pd.to_datetime(time_values[i - 1])
        t1 = pd.to_datetime(time_values[i])
        if t1 - t0 != pd.Timedelta(days=1):
            continue

        # Slice two time steps
        sample = input.isel(time=slice(i - 1, i + 1))  # shape (2, C, H, W)
        batch.append(sample)
        init_time_strs.append("".join(str(t1.date()).split('-')))

        # once batch_size is reached, yield the batch
        if len(batch) == batch_size:
            # stack into one batch
            yield xr.concat(batch, dim='batch'), init_time_strs
            batch = []
            init_time_strs = []

    # yield remaining batch if any
    if batch:
        yield xr.concat(batch, dim='batch'), init_time_strs


def main():
    start_date = '1990-01-01'
    end_date = '1990-12-31'
    wb2_ds = load_wb2_data('/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr')
    olr_ds = load_olr_data('/glade/derecho/scratch/kvirji/DATA/NOAA/OLR/PSL_interpolated/olr.day.mean.nc')
    u100_ds = load_cds_data('/glade/derecho/scratch/kvirji/DATA/100u')
    v100_ds = load_cds_data('/glade/derecho/scratch/kvirji/DATA/100v')
    # mask = xr.open_dataarray("/glade/derecho/scratch/kvirji/FuXi-S2S/data/mask.nc")

    model = "/glade/derecho/scratch/kvirji/mjo-predict/pretrained_weights/model-1.0/fuxi_s2s.onnx"
    device = "cuda"
    batch_size = 1
    total_steps = 60
    total_members = 51
    save_dir = "/glade/derecho/scratch/kvirji/DATA/MJO/FuXi/"

    wb2_ds = slice_to(wb2_ds, start_date, end_date)
    olr_ds = slice_to(olr_ds, start_date, end_date)
    u100_ds = slice_to(u100_ds, start_date, end_date)
    v100_ds = slice_to(v100_ds, start_date, end_date)

    olr_ds = regrid_to(olr_ds, wb2_ds.lat, wb2_ds.lon)
    u100_ds = regrid_to(u100_ds, wb2_ds.lat, wb2_ds.lon)
    v100_ds = regrid_to(v100_ds, wb2_ds.lat, wb2_ds.lon)

    wb2_ds['top_net_thermal_radiation'] = -olr_ds['olr']
    wb2_ds['100m_u_component_of_wind'] = u100_ds['u100']
    wb2_ds['100m_v_component_of_wind'] = v100_ds['v100']
    
    ds = clean(wb2_ds)
    input = format(ds)
    input = input.sel(lat=input.lat[::-1])

    print_dataarray(input)

    start = time.perf_counter()
    model = load_model(model, device)
    print(f'FuXi took {time.perf_counter() - start:.2f} sec to load.')

    for batch, time_strs in batch_input(input, batch_size=batch_size):
        print('Processing: ', time_strs)
        run_inference(
            model, 
            batch, 
            total_steps, 
            total_members,  
            time_strs,
            save_dir=save_dir
        )
  

if __name__ == "__main__":
    main()