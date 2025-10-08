import os
from tqdm import tqdm
from datetime import datetime
from mjo.utils.RMM.io import load_rmm_indices


def load_forecast(predict_dir, start_date, end_date, member=None):
    dataframes = []
    max_lt = -1
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = sorted([d for d in os.listdir(predict_dir) if start <= datetime.strptime(d, "%Y-%m-%d" if member else "%Y-%m-%d.txt") < end])
    for d in tqdm(dates, f'Loading data from {predict_dir}'):
        filepath = os.path.join(predict_dir, d, f'{member}.txt') if member else os.path.join(predict_dir, d)
        df = load_rmm_indices(filepath)
        max_lt = max(max_lt, len(df))
        dataframes.append(df)
    return dataframes, max_lt, dates