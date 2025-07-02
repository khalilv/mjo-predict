#!/usr/bin/env python3

'''
nohup python download_era5.py > download_era5.log 2>&1 &
2m_temperature
total_precipitation
surface_solar_radiation_downwards (ssrd)
'''

import os
import cdsapi
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the client
client = cdsapi.Client()

# Error handling and logging function
def download_data(year, variable, short_name):
    try:
        filename = f'{short_name}_{year}.nc'
        if not os.path.exists(filename):
            dataset = "derived-era5-single-levels-daily-statistics"
            request = {
                "product_type": ["reanalysis"],
                'variable': [variable],
                'year': [str(year)],
                'month': [str(m).zfill(2) for m in range(1,13)],
                'day': [str(d).zfill(2) for d in range(1,32)],
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "1_hourly"
            }
            temp_filename = filename + ".tmp"
            client.retrieve(dataset, request, temp_filename)
            os.rename(temp_filename, filename)  # Rename after successful download
            logging.info(f"Downloaded: {filename}")
        else:
            logging.info(f"File already exists: {filename}")
    except cdsapi.ClientException as e:
        logging.error(f"CDS API Client error for {year}: {e}")
    except Exception as e:
        logging.error(f"Failed to download data for {year}: {e}")

# ----------- specify parameters here -----------------
vtd = "100m_u_component_of_wind"
sname = "100u"

# Update the save directory
sdir = f'/glade/derecho/scratch/kvirji/DATA/{sname}/'
os.makedirs(sdir, exist_ok=True)
os.chdir(sdir)

# Main loop
for year in range(1979, 2025):
    download_data(year, vtd, sname)