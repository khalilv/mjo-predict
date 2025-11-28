# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from mjo.utils.datamodule import MJOForecastDataModule
from mjo.TSMixer.module import MJOForecastModule
from pytorch_lightning.cli import LightningCLI

def main():
    """Train TSMixer model for MJO forecasting.

    Configures and trains a TSMixer model with time and feature mixing using PyTorch Lightning.
    Configuration is loaded from a YAML file specified via command line.

    """
    # Initialize Lightning with the model and data modules
    cli = LightningCLI(
        model_class=MJOForecastModule,
        datamodule_class=MJOForecastDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    # Configure model based on datamodule settings
    cli.model.set_input_length(len(cli.datamodule.get_history()) + 1)
    cli.model.set_output_length(len(cli.datamodule.get_predictions()))
    cli.model.set_input_dim(len(cli.datamodule.get_in_variables()))
    cli.model.set_output_dim(len(cli.datamodule.get_out_variables()))
    cli.model.set_out_variables(cli.datamodule.get_out_variables())
    if cli.datamodule.normalize_data:
        cli.model.set_denormalization(cli.datamodule.get_transforms('out'))
    cli.model.init_metrics()
    cli.model.init_network()

    # Train the model
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # Evaluate on test set using best checkpoint
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path='best')

if __name__ == "__main__":
    main()
