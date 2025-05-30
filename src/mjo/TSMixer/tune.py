# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import optuna

from mjo.utils.datamodule import MJOForecastDataModule
from mjo.TSMixer.module import MJOForecastModule
from pytorch_lightning.cli import LightningCLI
from optuna_integration import PyTorchLightningPruningCallback

# 1) entry point high-level class for training TSMixer 

# Define the objective function for optuna
def objective(trial):
    # Set the hyperparameters to optimize
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256, 512])
    ff_size = trial.suggest_categorical('ff_size', [32, 64, 128, 256, 512])
    num_blocks = trial.suggest_categorical('num_blocks', [2, 4, 8, 16])
    lr = trial.suggest_categorical('lr', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.3, 0.5, 0.7, 0.9])
    
    cli = LightningCLI(
        model_class=MJOForecastModule,
        datamodule_class=MJOForecastDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

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
    
    # Optuna pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val/mse_norm')
        
    cli.trainer.callbacks.append(pruning_callback)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    return cli.trainer.callback_metrics['val/mse_norm'].item()

def run_optimization(n_trials=50):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10, interval_steps=2)    
    sampler = optuna.samplers.TPESampler(multivariate=True,group=True,seed=42)
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)        
    return study

def main():
    study = run_optimization(n_trials=50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Loss: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # # Visualize the results
    # try:
    #     # Plot optimization history
    #     optuna.visualization.plot_optimization_history(study)
        
    #     # Plot parameter importances
    #     optuna.visualization.plot_param_importances(study)
        
    #     # Plot parallel coordinate plot
    #     optuna.visualization.plot_parallel_coordinate(study)
    # except ImportError:
    #     print("Visualization requires plotly. Install with: pip install plotly")

if __name__ == "__main__":
    main()
