# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
import optuna

from mjo.utils.datamodule import MJOForecastDataModule
from mjo.DLBC.module import MJOForecastModule
from pytorch_lightning.cli import LightningCLI
from optuna_integration import PyTorchLightningPruningCallback

def objective(trial):
    """Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object for suggesting hyperparameters

    Returns:
        Validation MSE for the trial
    """
    # Suggest hyperparameters to optimize
    hidden_size = 2 ** trial.suggest_int('hidden_exp', 5, 10)  # 32 to 1024
    lr = trial.suggest_float('lr', 1e-7, 1e-3, log=True)

    # Initialize Lightning with the model and data modules
    cli = LightningCLI(
        model_class=MJOForecastModule,
        datamodule_class=MJOForecastDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )

    root_dir = cli.trainer.default_root_dir
    os.makedirs(root_dir, exist_ok=True)

    # Configure model with suggested hyperparameters
    cli.model.set_input_length(len(cli.datamodule.get_predictions()))
    cli.model.set_input_dim(len(cli.datamodule.get_in_variables()) + len(cli.datamodule.get_date_variables()))
    cli.model.set_out_variables(cli.datamodule.get_out_variables())
    if cli.datamodule.normalize_data:
        cli.model.set_denormalization(cli.datamodule.get_transforms('out'))
    cli.model.hidden_size = hidden_size
    cli.model.lr = lr
    cli.model.init_metrics()
    cli.model.init_network()

    # Add pruning callback to stop unpromising trials early
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val/mse')
    cli.trainer.callbacks.append(pruning_callback)

    # Train and return validation metric
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    return cli.trainer.callback_metrics['val/mse'].item()

def run_optimization(n_trials, root_dir):
    """Run Optuna hyperparameter optimization study.

    Args:
        n_trials: Number of optimization trials to run
        root_dir: Directory to save optimization database

    Returns:
        Tuple of (study, db_path) where study is the Optuna study object
        and db_path is the path to the SQLite database
    """
    # Configure pruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=20,
        n_min_trials=20
    )
    # Configure TPE sampler for efficient hyperparameter search
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        seed=42
    )

    # Create SQLite database for storing trial results
    db_path = os.path.join(root_dir, "dlbc_optuna.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        sampler=sampler,
        storage=storage,
        study_name="dlbc_hyperparam_tuning",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    return study, db_path

def main():
    """Run hyperparameter tuning for DLBC model using Optuna.

    Performs N trials of hyperparameter optimization using TPE sampling
    and median pruning. Best trial parameters are saved to best_trial.yaml.

    """
    # Create CLI to access configuration
    cli = LightningCLI(
        model_class=MJOForecastModule,
        datamodule_class=MJOForecastDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )

    root_dir = cli.trainer.default_root_dir
    os.makedirs(root_dir, exist_ok=True)

    # Run optimization study
    study, _ = run_optimization(n_trials=25, root_dir=root_dir)

    # Print and save best trial results
    trial = study.best_trial
    print("Best trial:")
    print(f"  Loss: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open(os.path.join(root_dir, "best_trial.yaml"), "w") as f:
        yaml.dump(trial.params, f)

if __name__ == "__main__":
    main()
