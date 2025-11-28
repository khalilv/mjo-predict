# MJO Forecasting with Deep Learning

Framework to forcast the Madden-Julian Oscillation (MJO) using deep learning models.

## Installation

```bash
# Clone repository
git clone <repository-url>
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

# Monthly error analysis
python src/mjo/utils/analysis/monthly_errors.py --config MY_CONFIG
```

## Repository Structure

```
mjo-predict/
├── configs/          # YAML configuration files
├── src/mjo/
│   ├── DLBC/        # DLBC model
│   ├── TFT/         # TFT model
│   ├── TSMixer/     # TSMixer model
│   └── utils/       # Data processing, metrics, analysis
├── environment.yml  # Conda environment
└── pyproject.toml   # Package metadata
```

## Features

- **Models**: TSMixer, TFT, DLBC architectures
- **Training**: PyTorch Lightning with distributed GPU support
- **Optimization**: Optuna for hyperparameter tuning
- **RMM Processing**: RMM index calculation for observations and forecast data
- **Metrics**: MSE, MAE, BMSE, BCORR
- **Analysis**: Lead time heatmaps, lead time errors, monthly errors, filtering on initial and forecasted amplitude/phase
