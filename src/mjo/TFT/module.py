# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
import os
import math
import torch
from typing import Any, Dict, Union
from pytorch_lightning import LightningModule
from mjo.TFT.model import TFTModel
from mjo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from mjo.utils.metrics import MSE, MAE
from mjo.utils.RMM.io import save_rmm_indices
from mjo.utils.data_utils import prep_input

class MJOForecastModule(LightningModule):
    """
    PyTorch Lightning module for MJO (Madden-Julian Oscillation) forecasting using the Temporal Fusion Transformer (TFT) model.

    This module encapsulates the TFT architecture and training logic for MJO forecasting tasks.
    It supports flexible configuration of model architecture, optimization, and training schedule.

    Args:
        pretrained_path (str, optional): Path to pre-trained checkpoint. Default: "".
        num_static_components (int, optional): Number of static components (not variables) of the input. Default: 0.
        hidden_size (int, optional): Hidden state size of the TFT. Default: 64.
        lstm_layers (int, optional): Number of LSTM layers. Default: 1.
        num_attention_heads (int, optional): Number of attention heads. Default: 4.
        full_attention (bool, optional): Whether to use full attention. Default: False.
        feed_forward (str, optional): Feedforward block type. Default: "GatedResidualNetwork".
        hidden_continuous_size (int, optional): Hidden size for continuous variables. Default: 16.
        categorical_embedding_sizes (Dict, optional): Embedding sizes for categorical variables. Default: None (interpreted as {}).
        add_relative_index (bool, optional): Whether to add relative index. Default: True.
        dropout (float, optional): Dropout rate. Default: 0.1.
        norm_type (Union[str, torch.nn.Module], optional): Type of normalization to use. Default: "LayerNorm".
        lr (float, optional): Learning rate. Default: 5e-4.
        beta_1 (float, optional): Beta 1 for AdamW optimizer. Default: 0.9.
        beta_2 (float, optional): Beta 2 for AdamW optimizer. Default: 0.99.
        weight_decay (float, optional): Weight decay for AdamW optimizer. Default: 1e-5.
        warmup_steps (int, optional): Number of warmup steps for learning rate scheduler. Default: 1000.
        max_steps (int, optional): Maximum number of steps for learning rate scheduler. Default: 50000.
        save_outputs (bool, optional): Whether to save model outputs. Default: False.
    """

    def __init__(
        self,
        pretrained_path: str = "",
        num_static_components: int = 0,
        hidden_size: int = 64,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        full_attention: bool = False,
        feed_forward: str = "GatedResidualNetwork",
        hidden_continuous_size: int = 16,
        categorical_embedding_sizes: Dict = None,
        add_relative_index: bool = True,
        dropout: float = 0.1,
        norm_type: Union[str, torch.nn.Module] = "LayerNorm",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        save_outputs: bool = False
    ):
        super().__init__()
        # Model architecture parameters
        self.pretrained_path = pretrained_path
        self.num_static_components = num_static_components
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.full_attention = full_attention
        self.feed_forward = feed_forward
        self.hidden_continuous_size = hidden_continuous_size
        self.categorical_embedding_sizes = categorical_embedding_sizes if categorical_embedding_sizes is not None else {}
        self.add_relative_index = add_relative_index
        self.dropout = dropout
        self.norm_type = norm_type

        # Optimization and scheduler parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        # Output and utility attributes
        self.save_outputs = save_outputs
        self.denormalization = None

        #placeholders
        self.in_variables = []
        self.date_variables = []
        self.out_variables = []
        self.forecast_variables = []

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

    def set_in_variables(self, in_variables: list):
        self.in_variables = in_variables
    
    def set_forecast_variables(self, forecast_variables: list):
        self.forecast_variables = forecast_variables

    def set_input_length(self, input_length: int):
        self.input_chunk_length = input_length

    def set_output_length(self, output_length: int):
        self.output_chunk_length = output_length
    
    def set_date_variables(self, date_variables: list):
        self.date_variables = date_variables

    def set_out_variables(self, out_variables: list):
        self.out_variables = out_variables
        self.output_dim = (len(self.out_variables), 1)
    
    def init_metrics(self):
        denormalize = self.denormalization.denormalize if self.denormalization else None

        self.train_mse = MSE(vars=self.out_variables, transforms=None, suffix='norm')
        self.train_mae = MAE(vars=self.out_variables, transforms=None, suffix='norm')
        self.val_mse = MSE(vars=self.out_variables, transforms=denormalize, suffix=None)
        self.val_mae = MAE(vars=self.out_variables, transforms=denormalize, suffix=None)        
        self.test_mse = MSE(vars=self.out_variables, transforms=denormalize, suffix=None)
        self.test_mae = MAE(vars=self.out_variables, transforms=denormalize, suffix=None)

    def init_network(self):
        self.variables_meta = {
            "model_config": {
                "reals_input": self.in_variables + self.date_variables + self.forecast_variables,
                "time_varying_encoder_input": self.in_variables + self.date_variables,
                "time_varying_decoder_input": self.forecast_variables + self.date_variables,
                "static_input": [],
                "static_input_numeric": [],
                "static_input_categorical": [],
                "categorical_input": []
            }
        }
        self.net = TFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            output_dim=self.output_dim,
            variables_meta=self.variables_meta,
            num_static_components=self.num_static_components,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            num_attention_heads=self.num_attention_heads,
            full_attention=self.full_attention,
            feed_forward=self.feed_forward,
            hidden_continuous_size=self.hidden_continuous_size,
            categorical_embedding_sizes=self.categorical_embedding_sizes,
            add_relative_index=self.add_relative_index,
            dropout=self.dropout,
            norm_type=self.norm_type
        )
        if hasattr(self, "pretrained_path") and self.pretrained_path and len(self.pretrained_path) > 0:
            self.load_pretrained_weights(self.pretrained_path)

    def set_denormalization(self, denormalization):
        self.denormalization = denormalization
    
    def setup(self, stage: str):
        if self.denormalization:
            self.denormalization.to(device=self.device, dtype=self.dtype)
    
    def training_step(self, batch: Any, batch_idx: int):
        in_data, in_date_encodings, out_data, out_date_encodings, forecast_data, residual, in_variables, date_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps = batch

        target = residual if residual is not None else out_data

        x_in = prep_input(
            in_data=in_data,
            in_date_encodings=in_date_encodings,
            out_date_encodings=out_date_encodings,
            forecast_data=forecast_data,
            out_timestamps=out_timestamps,
            forecast_timestamps=forecast_timestamps,
        )       
        pred_data = self.net.forward(x_in=x_in)
        pred_data = pred_data.squeeze(dim=-1)


        batch_loss_mse = self.train_mse(preds=pred_data, targets=target)
        batch_loss_mae = self.train_mae(preds=pred_data, targets=target)
        batch_loss = {**batch_loss_mae, **batch_loss_mse}

        for key in batch_loss.keys():
            self.log(
                "train/" + key,
                batch_loss[key],
                prog_bar=True,
            )

        self.train_mse.reset()
        return batch_loss['mse_norm'] #batch_loss['mae_norm']
    
    def validation_step(self, batch: Any, batch_idx: int):
        in_data, in_date_encodings, out_data, out_date_encodings, forecast_data, residual, in_variables, date_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps = batch

        target = residual if residual is not None else out_data

        x_in = prep_input(
            in_data=in_data,
            in_date_encodings=in_date_encodings,
            out_date_encodings=out_date_encodings,
            forecast_data=forecast_data,
            out_timestamps=out_timestamps,
            forecast_timestamps=forecast_timestamps,
        ) 
        
        pred_data = self.net.forward(x_in=x_in)
        pred_data = pred_data.squeeze(dim=-1)

        self.val_mse.update(preds=pred_data, targets=target)
        self.val_mae.update(preds=pred_data, targets=target)
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
        in_data, in_date_encodings, out_data, out_date_encodings, forecast_data, residual, in_variables, date_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps = batch

        target = residual if residual is not None else out_data

        x_in = prep_input(
            in_data=in_data,
            in_date_encodings=in_date_encodings,
            out_date_encodings=out_date_encodings,
            forecast_data=forecast_data,
            out_timestamps=out_timestamps,
            forecast_timestamps=forecast_timestamps,
        ) 

        pred_data = self.net.forward(x_in=x_in)
        pred_data = pred_data.squeeze(dim=-1)

        self.test_mse.update(preds=pred_data, targets=target)
        self.test_mae.update(preds=pred_data, targets=target)

        if self.save_outputs:
            if forecast_data is not None:
                pred_data = pred_data + out_data - residual #residual + forecast to recover prediction
            pred_data = self.denormalization.denormalize(pred_data)
            pred_data = pred_data.cpu().numpy()
            for b in range(pred_data.shape[0]):
                filename = f'{str(in_timestamps[b][-1]).split("T")[0]}.txt'
                save_rmm_indices(
                    time=out_timestamps[b],
                    RMM1=pred_data[b,:,out_variables.index('RMM1')],
                    RMM2=pred_data[b,:,out_variables.index('RMM2')],
                    filename=os.path.join(self.output_dir, filename),
                    method_str='TFT'
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
