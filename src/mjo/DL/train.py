# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from omegaconf import OmegaConf
from mjo.utils.datamodule import MJOForecastDataModule
from mjo.DL.module import MJOForecastModule
from pytorch_lightning.cli import LightningCLI

# 1) entry point high-level class for training bias correction modules

def main():
    phase_groups = [[1,8], [2,3], [4,5], [6,7]]

    for group in phase_groups:
        group_suffix = f"p{''.join(map(str, group))}"
        args = sys.argv[1:]

        # Intercept --config or default config path
        if '--config' in args:
            config_idx = args.index('--config') + 1
            config_path = args[config_idx]
        else:
            raise ValueError("No config file provided. Please specify a config file using the '--config' argument.")

        config = OmegaConf.load(config_path)
        base_dir = config.trainer.default_root_dir
        group_dir = os.path.join(base_dir, group_suffix)
        sys.argv = [
                sys.argv[0], 
                '--config', config_path,
                f'--trainer.default_root_dir={group_dir}',
                f'--data.filter_mjo_phases=[{",".join(map(str, group))}]'
                ]
        
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
