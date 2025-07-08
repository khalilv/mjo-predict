# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
import os
import math
import torch
from typing import Any
from pytorch_lightning import LightningModule
from mjo.TSMixer.model import TSMixerX
from mjo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from mjo.utils.metrics import MSE, MAE
from mjo.utils.RMM.io import save_rmm_indices
from mjo.utils.data_utils import prep_input

class MJOForecastModule(LightningModule):
    """
    PyTorch Lightning module for MJO (Madden-Julian Oscillation) forecasting using the TSMixer model.

    This module encapsulates the TSMixer architecture and training logic for MJO forecasting tasks.
    It supports flexible configuration of model architecture, optimization, and training schedule.

    Args:
        pretrained_path (str): Path to pre-trained checkpoint. Default: "".
        past_cov_dim (int): Number of past covariate features. Default: 0.
        future_cov_dim (int): Number of future covariate features. Default: 0.
        static_cov_dim (int): Number of static covariate features. Default: 0.
        nr_params (int): Number of parameters for the likelihood (or 1 if not probabilistic). Default: 1.
        hidden_size (int): Hidden state size of the TSMixer. Default: 64.
        ff_size (int): Dimension of the feedforward network internal to the module. Default: 64.
        num_blocks (int): Number of mixer blocks. Default: 2.
        activation (str): Activation function to use. Default: 'ReLU'.
        dropout (float): Dropout rate for regularization. Default: 0.1.
        norm_type (str): Type of normalization to use. Default: 'TimeBatchNorm2d'.
        normalize_before (bool): Whether to apply normalization before or after mixing. Default: True.
        lr (float): Learning rate. Default: 5e-4.
        beta_1 (float): Beta 1 for AdamW optimizer. Default: 0.9.
        beta_2 (float): Beta 2 for AdamW optimizer. Default: 0.99.
        weight_decay (float): Weight decay for AdamW optimizer. Default: 1e-5.
        warmup_steps (int): Number of warmup steps for learning rate scheduler. Default: 1000.
        max_steps (int): Maximum number of steps for learning rate scheduler. Default: 50000.
        save_outputs (bool): Whether to save model outputs. Default: False.
    """

    def __init__(
        self,
        pretrained_path: str = "",      
        past_cov_dim: int = 0,
        future_cov_dim: int = 0,
        static_cov_dim: int = 0,
        nr_params: int = 1,
        hidden_size: int = 64,
        ff_size: int = 64,
        num_blocks: int = 2,
        activation: str = 'ReLU',
        dropout: float = 0.1,
        norm_type: str = 'TimeBatchNorm2d',
        normalize_before: bool = True,
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        load_samples_without_forecasts_until_epoch: int = 0,
        save_outputs: bool = False
    ):
        super().__init__()
        # Model architecture parameters
        self.pretrained_path = pretrained_path
        self.past_cov_dim = past_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        self.num_blocks = num_blocks
        self.activation = activation
        self.dropout = dropout
        self.norm_type = norm_type
        self.normalize_before = normalize_before

        # Optimization and scheduler parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.load_samples_without_forecasts_until_epoch = load_samples_without_forecasts_until_epoch

        # Output and utility attributes
        self.save_outputs = save_outputs
        self.denormalization = None
        self.year_normalization = None

        self.save_hyperparameters(logger=False, ignore=["net"])      

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location=torch.device("cpu"), weights_only=True)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"), weights_only=True)
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=True)
        print(msg)

    def set_input_length(self, input_length: int):
        self.input_length = input_length
    
    def set_output_length(self, output_length: int):
        self.output_length = output_length
    
    def set_input_dim(self, input_dim: int):
        self.input_dim = input_dim
    
    def set_output_dim(self, output_dim: int):
        self.output_dim = output_dim
    
    def set_out_variables(self, out_variables: list):
        self.out_variables = out_variables

    def init_metrics(self):
        denormalize = self.denormalization.denormalize if self.denormalization else None

        self.train_mse = MSE(vars=self.out_variables, transforms=None, suffix='norm')
        self.train_mae = MAE(vars=self.out_variables, transforms=None, suffix='norm')
        self.val_mse = MSE(vars=self.out_variables, transforms=denormalize, suffix=None)
        self.val_mae = MAE(vars=self.out_variables, transforms=denormalize, suffix=None)        
        self.test_mse = MSE(vars=self.out_variables, transforms=denormalize, suffix=None)
        self.test_mae = MAE(vars=self.out_variables, transforms=denormalize, suffix=None)

    def init_network(self):
        self.net = TSMixerX(
            input_length=self.input_length,
            output_length=self.output_length,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            past_cov_dim=self.past_cov_dim,
            future_cov_dim=self.future_cov_dim,
            static_cov_dim=self.static_cov_dim,
            nr_params=self.nr_params,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            num_blocks=self.num_blocks,
            activation=self.activation,
            dropout=self.dropout,
            norm_type=self.norm_type,
            normalize_before=self.normalize_before,
        )
        if hasattr(self, "pretrained_path") and self.pretrained_path and len(self.pretrained_path) > 0:
            self.load_pretrained_weights(self.pretrained_path)
            
    def set_denormalization(self, denormalization):
        self.denormalization = denormalization
    
    def set_year_normalization(self, normalization):
        self.year_normalization = normalization
    
    def setup(self, stage: str):
        if self.denormalization:
            self.denormalization.to(device=self.device, dtype=self.dtype)
        if self.year_normalization:
            self.year_normalization.to(device=self.device, dtype=self.dtype)
    
    def on_train_epoch_end(self) -> None:
        p = self.get_load_samples_without_forecast_prob(self.current_epoch + 1)
        self.trainer.datamodule.data_train.dataset.load_samples_without_forecast_prob = p
        self.log('train/load_samples_without_forecast_prob', p)
        return

    def get_load_samples_without_forecast_prob(self, epoch):
        if self.load_samples_without_forecasts_until_epoch <= 0:
            return 0.0

        # linear decay from 1.0 to 0.0
        linear_decay = 1.0 - (epoch / self.load_samples_without_forecasts_until_epoch)
        return max(0.0, min(1.0, linear_decay))
        
    def training_step(self, batch: Any, batch_idx: int):
        in_data, out_data, forecast_data, forecast_mask, in_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps = batch

        x_in = prep_input(
            in_data=in_data,
            forecast_data=forecast_data,
            forecast_mask=forecast_mask,
            in_timestamps=in_timestamps,
            out_timestamps=out_timestamps,
            forecast_timestamps=forecast_timestamps,
            device=self.device,
            dtype=self.dtype,
            year_normalization=self.year_normalization
        ) 
        
        pred_data = self.net.forward(x_in=x_in)
        pred_data = pred_data.squeeze(dim=-1)

        batch_loss_mse = self.train_mse(preds=pred_data, targets=out_data)
        batch_loss_mae = self.train_mae(preds=pred_data, targets=out_data)
        batch_loss = {**batch_loss_mae, **batch_loss_mse}

        for key in batch_loss.keys():
            self.log(
                "train/" + key,
                batch_loss[key],
                prog_bar=True,
            )

        self.train_mse.reset()
        self.train_mae.reset()
        return batch_loss['mae_norm']
    
    def validation_step(self, batch: Any, batch_idx: int):
        in_data, out_data, forecast_data, forecast_mask, in_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps = batch

        x_in = prep_input(
            in_data=in_data,
            forecast_data=forecast_data,
            forecast_mask=forecast_mask,
            in_timestamps=in_timestamps,
            out_timestamps=out_timestamps,
            forecast_timestamps=forecast_timestamps,
            device=self.device,
            dtype=self.dtype,
            year_normalization=self.year_normalization
        ) 
        
        pred_data = self.net.forward(x_in=x_in)
        pred_data = pred_data.squeeze(dim=-1)

        self.val_mse.update(preds=pred_data, targets=out_data)
        self.val_mae.update(preds=pred_data, targets=out_data)
        return
        
    def on_validation_epoch_end(self):
        val_mse = self.val_mse.compute()
        val_mae = self.val_mae.compute()
               
        #scalar metrics
        loss_dict = {**val_mse, **val_mae}
        for key in loss_dict.keys():
            self.log(
                "val/" + key,
                loss_dict[key],
                prog_bar=True,
                sync_dist=True
            )
        self.val_mse.reset()
        self.val_mae.reset()
        return loss_dict
    
    def on_test_epoch_start(self):
        if self.save_outputs:
            self.output_dir = f'{self.logger.log_dir}/outputs/'
            os.makedirs(self.output_dir, exist_ok=False)

    def test_step(self, batch: Any, batch_idx: int):
        in_data, out_data, forecast_data, forecast_mask, in_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps = batch
       
        x_in = prep_input(
            in_data=in_data,
            forecast_data=forecast_data,
            forecast_mask=forecast_mask,
            in_timestamps=in_timestamps,
            out_timestamps=out_timestamps,
            forecast_timestamps=forecast_timestamps,
            device=self.device,
            dtype=self.dtype,
            year_normalization=self.year_normalization
        ) 

        pred_data = self.net.forward(x_in=x_in)
        pred_data = pred_data.squeeze(dim=-1)

        self.test_mse.update(preds=pred_data, targets=out_data)
        self.test_mae.update(preds=pred_data, targets=out_data)
       
        if self.save_outputs:
            pred_data = self.denormalization.denormalize(pred_data)
            pred_data = pred_data.cpu().numpy()
            for b in range(pred_data.shape[0]):
                filename = f'{str(in_timestamps[b][-1]).split("T")[0]}.txt'
                save_rmm_indices(
                    time=out_timestamps[b],
                    RMM1=pred_data[b,:,out_variables.index('RMM1')],
                    RMM2=pred_data[b,:,out_variables.index('RMM2')],
                    filename=os.path.join(self.output_dir, filename),
                    method_str='TSMixer'
                )
        return
    
    def on_test_epoch_end(self):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
               
        #scalar metrics
        loss_dict = {**test_mse, **test_mae}
        for key in loss_dict.keys():
            self.log(
                "test/" + key,
                loss_dict[key],
                prog_bar=False,
                sync_dist=True
            )
        self.test_mse.reset()
        self.test_mae.reset()
        return loss_dict

    #optimizer definition - will be used to optimize the network
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta_1, self.beta_2),
            weight_decay=self.weight_decay,
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.warmup_steps,
            self.max_steps,
            warmup_start_lr=self.lr/10,
            eta_min=self.lr/10,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
