# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
import optuna

from mjo.utils.datamodule import MJOForecastDataModule
from mjo.TFT.module import MJOForecastModule
from pytorch_lightning.cli import LightningCLI
from optuna_integration import PyTorchLightningPruningCallback

def objective(trial):
    
    # hyperparameters to optimize
    hidden_size = 2 ** trial.suggest_int('hidden_exp', 5, 10)
    # lstm_layers = trial.suggest_int('lstm_layers', 1, 4)
    # num_attention_heads = trial.suggest_int('num_attention_heads', 2, 16, step=2)
    # hidden_continuous_size = 2 ** trial.suggest_int('hidden_continuous_exp', 5, 10)
    # add_relative_index = trial.suggest_categorical('add_relative_index', [True, False])
    # full_attention = trial.suggest_categorical('full_attention', [True, False])
    dropout = trial.suggest_float('dropout', 0.1, 0.9, step=0.1)
    lr = trial.suggest_float('lr', 1e-7, 1e-3, log=True)
    # beta_1 = trial.suggest_float("beta_1", 0.85, 0.99, step=0.01)
    # beta_2 = trial.suggest_float("beta_2", 0.95, 0.999, step=0.01)
    # weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)

    # Set up CLI (without running)
    cli = LightningCLI(
        model_class=MJOForecastModule,
        datamodule_class=MJOForecastDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )

    # ensure root dir exists
    root_dir = cli.trainer.default_root_dir
    os.makedirs(root_dir, exist_ok=True)

    cli.model.set_input_length(len(cli.datamodule.get_history()) + 1)
    cli.model.set_output_length(len(cli.datamodule.get_predictions()))
    cli.model.set_in_variables(cli.datamodule.get_in_variables())
    cli.model.set_date_variables(cli.datamodule.get_date_variables())
    cli.model.set_out_variables(cli.datamodule.get_out_variables())
    cli.model.set_forecast_variables(cli.datamodule.get_forecast_variables())
    if cli.datamodule.normalize_data:
        cli.model.set_denormalization(cli.datamodule.get_transforms('out'))
    cli.model.init_metrics()

    cli.model.hidden_size = hidden_size
    # # cli.model.lstm_layers = lstm_layers
    # # cli.model.num_attention_heads = num_attention_heads
    # cli.model.hidden_continuous_size = hidden_size
    # cli.model.add_relative_index = add_relative_index
    # cli.model.full_attention = full_attention
    cli.model.lr = lr
    cli.model.dropout = dropout
    # cli.model.beta_1 = beta_1
    # cli.model.beta_2 = beta_2
    # cli.model.weight_decay = weight_decay
    cli.model.init_network()

    # pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val/mse')
    cli.trainer.callbacks.append(pruning_callback)

    # train
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    return cli.trainer.callback_metrics['val/mse'].item()

def run_optimization(n_trials, root_dir):
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=20,
        n_min_trials=20
    )
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        seed=42
    )

    # Path to DB file inside root_dir
    db_path = os.path.join(root_dir, "tft_optuna.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        sampler=sampler,
        storage=storage,
        study_name="tft_hyperparam_tuning",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    return study, db_path

def main():
    # Create CLI once to access root_dir before optuna runs
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

    # Run optimization
    study, _ = run_optimization(n_trials=50, root_dir=root_dir)

    # Save best trial params to YAML in root_dir
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
