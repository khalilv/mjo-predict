# MJO Forecasting with Deep Learning

Deep learning models for Madden-Julian Oscillation (MJO) forecasting over sub-seasonal to seasonal horizons.

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

**TSMixer** - Time and feature mixing for multivariate forecasting
**TFT** - Temporal Fusion Transformer with multi-head attention
**DLBC** - Deep Learning Bias Correction for dynamical forecasts

## Usage

### Training
```bash
# Train models
python src/mjo/TSMixer/train.py --config configs/TSMixer/train.yaml
python src/mjo/TFT/train.py --config configs/TFT/train.yaml
python src/mjo/DLBC/train.py --config configs/DLBC/train.yaml
```

### Hyperparameter Tuning
```bash
python src/mjo/TSMixer/tune.py --config configs/TSMixer/tune.yaml
```

### RMM Processing
```bash
# Process RMM indices from various sources
python src/mjo/utils/RMM/ecmwf.py        # ECMWF forecast data
python src/mjo/utils/RMM/mean.py          # Ensemble mean processing
python src/mjo/utils/RMM/eof.py           # EOF-based RMM calculation

# FuXi model processing
python src/mjo/utils/RMM/FuXi/preprocess.py  # Preprocess FuXi data
python src/mjo/utils/RMM/FuXi/forecast.py    # Generate forecasts
python src/mjo/utils/RMM/FuXi/generate.py    # Generate RMM indices
```

### Analysis
```bash
# Generate lead time performance heatmaps
python src/mjo/utils/analysis/lead_time_heatmaps.py --config configs/analysis/lead_time_heatmaps.yaml

# Monthly error analysis
python src/mjo/utils/analysis/monthly_errors.py --config configs/analysis/monthly_errors.yaml
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
- **Training**: PyTorch Lightning with GPU support
- **Optimization**: Optuna for hyperparameter tuning
- **RMM Processing**: ECMWF, FuXi, and EOF-based RMM index calculation
- **Metrics**: MSE, MAE, BMSE (bivariate with amplitude/phase)
- **Analysis**: Lead time heatmaps, monthly errors, skill vs. amplitude/phase
- **Data**: NPZ format for RMM indices and forecast data
