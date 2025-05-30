# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
import optuna

from mjo.utils.datamodule import MJOForecastDataModule
from mjo.TSMixer.module import MJOForecastModule
from pytorch_lightning.cli import LightningCLI
from optuna_integration import PyTorchLightningPruningCallback

def objective(trial):
    
    # hyperparameters to optimize
    hidden_size = 2 ** trial.suggest_int('hidden_exp', 5, 9)
    ff_size = 2 ** trial.suggest_int('ff_exp', 5, 9)
    num_blocks = trial.suggest_int('num_blocks', 2, 16)
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.9)

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

    # set model hyperparameters
    cli.model.set_input_length(len(cli.datamodule.get_history()) + 1)
    cli.model.set_output_length(len(cli.datamodule.get_predictions()))
    cli.model.set_input_dim(len(cli.datamodule.get_in_variables()))
    cli.model.set_output_dim(len(cli.datamodule.get_out_variables()))
    cli.model.set_out_variables(cli.datamodule.get_out_variables())

    if cli.datamodule.normalize_data:
        cli.model.set_denormalization(cli.datamodule.get_transforms('out'))

    cli.model.init_metrics()
    cli.model.hidden_size = hidden_size
    cli.model.ff_size = ff_size
    cli.model.num_blocks = num_blocks
    cli.model.lr = lr
    cli.model.dropout = dropout
    cli.model.init_network()

    # pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val/mse_norm')
    cli.trainer.callbacks.append(pruning_callback)

    # train
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    return cli.trainer.callback_metrics['val/mse_norm'].item()

def run_optimization(n_trials, root_dir):
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10,
        interval_steps=2
    )
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        seed=42
    )

    # Path to DB file inside root_dir
    db_path = os.path.join(root_dir, "tsmixer_optuna.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        sampler=sampler,
        storage=storage,
        study_name="tsmixer_hyperparam_tuning",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    return study, db_path

def main():
    # Create CLI once to access root_dir before Optuna runs
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
    study, _ = run_optimization(n_trials=100, root_dir=root_dir)

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
