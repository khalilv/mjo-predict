# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from mjo.utils.datamodule import MJOForecastDataModule
from mjo.DL.module import MJOForecastModule
from pytorch_lightning.cli import LightningCLI

# 1) entry point high-level class for training bias correction modules

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=MJOForecastModule,
        datamodule_class=MJOForecastDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )

    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    cli.model.set_input_length(len(cli.datamodule.get_predictions()))
    cli.model.set_input_dim(len(cli.datamodule.get_in_variables()))
    cli.model.set_out_variables(cli.datamodule.get_out_variables())
    if cli.datamodule.normalize_data:
        cli.model.set_denormalization(cli.datamodule.get_transforms('out'))
    cli.model.init_metrics()
    cli.model.init_network()

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path='best')

        
        
if __name__ == "__main__":
    main()
