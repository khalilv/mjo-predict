import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.metrics import mape
from darts.models import TFTModel
from darts.utils.likelihood_models.torch import QuantileRegression
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from mjo.utils.RMM.io import load_rmm_indices
warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)
# ----------------------------- Configuration -----------------------------
DATA_PATH = "/Users/kvirji/Documents/mjo-predict/DATA/MJO/RMM/reference_period_1979-09-07_to_2001-12-31/rmm.txt"
ROOT_DIR = "/Users/kvirji/Documents/mjo-predict/exps/TFT/historical_plus_dates/test"
MAX_ENCODER_LENGTH = 720
MAX_PREDICTION_LENGTH = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
ATTN_HEADS = 4
DROPOUT = 0.2
HIDDEN_CONT_SIZE = 32
LSTM_LAYERS = 1

# ----------------------------- Data Preparation -----------------------------
data = load_rmm_indices(filepath=DATA_PATH)
data = data.drop(columns=['phase'])

data_series = TimeSeries.from_dataframe(data)

train_end = pd.Timestamp("20190101")
val_start = train_end - pd.Timedelta(days=MAX_ENCODER_LENGTH)
val_end = pd.Timestamp("20210101")
test_start = val_end - pd.Timedelta(days=MAX_ENCODER_LENGTH)

train, _ = data_series.split_before(train_end)
_, val = data_series.split_before(val_start)
val, _ = val.split_before(val_end)
_, test = data_series.split_before(test_start)


transformer = Scaler()
train_transformed = transformer.fit_transform(train).astype(np.float32)
val_transformed = transformer.transform(val).astype(np.float32)
data_series_transformed = transformer.transform(data_series).astype(np.float32)

# create year, month and integer index covariate series
covariates = datetime_attribute_timeseries(data_series, attribute="year", one_hot=False)
covariates = covariates.stack(
    datetime_attribute_timeseries(data_series, attribute="month", one_hot=False)
)
covariates = covariates.stack(
    datetime_attribute_timeseries(data_series, attribute="day", one_hot=False)
)

covariates = covariates.astype(np.float32)
scaler_covs = Scaler()
cov_train, _ = covariates.split_before(train_end)
scaler_covs.fit(cov_train)
covariates_transformed = scaler_covs.transform(covariates)
print('done')

# default quantiles for QuantileRegression
quantiles = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]
tft = TFTModel(
    input_chunk_length=MAX_ENCODER_LENGTH,
    output_chunk_length=MAX_PREDICTION_LENGTH,
    hidden_size=HIDDEN_SIZE,
    hidden_continuous_size=HIDDEN_CONT_SIZE,
    lstm_layers=LSTM_LAYERS,
    num_attention_heads=ATTN_HEADS,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    n_epochs=2,
    add_relative_index=False,
    add_encoders=None,
    likelihood=QuantileRegression(
        quantiles=quantiles
    ),  
    random_state=42,
)

tft.fit(train_transformed, future_covariates=covariates_transformed, verbose=True)

def eval_model(model, n, actual_series, val_series):
    pred_series = model.predict(n=n, num_samples=200)

    # plot actual series
    plt.figure(figsize=(9,6))
    actual_series[: pred_series.end_time()].plot(label="actual")

    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=0.01, high_quantile=0.99, label='1-99th percentiles'
    )
    pred_series.plot(low_quantile=0.1, high_quantile=0.9, label='10-90th percentiles')

    plt.title(f"MAPE: {mape(val_series, pred_series):.2f}%")
    plt.legend()
    plt.savefig('test.png', dpi=300)


eval_model(tft, MAX_PREDICTION_LENGTH, data_series_transformed, val_transformed)
print('done')
