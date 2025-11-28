# MJO Forecasting with Deep Learning

Framework to forecast the Madden-Julian Oscillation (MJO) using deep learning models.

## Installation

```bash
# Clone repository
git clone https://github.com/khalilv/mjo-predict.git
cd mjo-predict

# Create environment
conda env create -f environment.yml
conda activate mjo
pip install -e .
```

## Models

**TSMixer** - Time and feature mixing model for multivariate forecasting

**TFT** - Temporal Fusion Transformer with multi-head attention

**DLBC** - Deep learning bias correction to correct MJO biases in forecasting models

## Usage

### Training
```bash
# Train models
python src/mjo/TSMixer/train.py --config MY_CONFIG
python src/mjo/TFT/train.py --config MY_CONFIG
python src/mjo/DLBC/train.py --config MY_CONFIG
```

### Hyperparameter Tuning
```bash
python src/mjo/TSMixer/tune.py --config MY_CONFIG
```

### RMM Computation
```bash
# Compute RMM indices from various sources
python src/mjo/utils/RMM/ecmwf.py        # ECMWF forecast data
python src/mjo/utils/RMM/observation.py  # Compute observed RMM indices
python src/mjo/utils/RMM/eof.py          # Compute EOF over a given reference period
python src/mjo/utils/RMM/preprocess.py   # Preprocess RMM indices for model training
python src/mjo/utils/RMM/io.py           # Handles saving/loading functionality

# FuXi-S2S model processing
python src/mjo/utils/RMM/FuXi/generate.py    # Generate probabilistic global forecasts with FuXi-S2S
python src/mjo/utils/RMM/FuXi/mean.py        # Compute FuXi-S2S ensemble mean forecast
python src/mjo/utils/RMM/FuXi/forecast.py    # Generate RMM indices from FuXi-S2S forecasts
python src/mjo/utils/RMM/FuXi/preprocess.py  # Preprocess FuXi-S2S RMM indices for model training
```

### Analysis
```bash
# Generate lead time performance heatmaps
python src/mjo/utils/analysis/lead_time_heatmaps.py --config MY_CONFIG

# Generate lead time BCORR/BMSE error plots
python src/mjo/utils/analysis/lead_time_errors.py --config MY_CONFIG

# Monthly error analysis
python src/mjo/utils/analysis/monthly_errors.py --config MY_CONFIG
```

## Data

The repository includes **FuXi-S2S MJO forecast data** stored in `data/`:
- `FuXi_2002-2011.zip` - MJO forecasts from 2002-2011
- `FuXi_2012-2021.zip` - MJO forecasts from 2012-2021

**Structure**: Each zip file contains forecast initialization dates organized as `FuXi/YYYY-MM-DD/`, with 51 ensemble members (00.txt through 50.txt) per date. Each member file contains the RMM forecast trajectory in text format with columns:

```
Year  Month  Day  RMM1  RMM2  Phase  Amplitude Method
```

The models use **RMM (Real-time Multivariate MJO) indices** to characterize the MJO. RMM indices are two time-varying components (RMM1, RMM2) derived from OLR (Outgoing Longwave Radiation) and zonal wind data at 850 hPa and 200 hPa. From RMM1 and RMM2, the MJO amplitude and phase (1-8) are quantified.


## Repository Structure

```
mjo-predict/
├── configs/          # YAML configuration files
├── data/             # FuXi-S2S RMM forecast data
├── src/mjo/
│   ├── DLBC/        # DLBC model
│   ├── TFT/         # TFT model
│   ├── TSMixer/     # TSMixer model
│   └── utils/       # Data processing, metrics, analysis
├── environment.yml  # Conda environment
└── pyproject.toml   # Package metadata
```
