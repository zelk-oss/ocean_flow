#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

# System modules
import logging
import abc
from typing import Any, Callable, Dict, Iterable

# External modules
import lightning.pytorch as pl
import torch

# Internal modules
from .utils import split_wd_params
from ocean_flow.pipelines import PrePipeline, PostPipeline


main_logger = logging.getLogger(__name__)


# Flow Matching module 
class TrainingModule(pl.LightningModule):
    """
    Flow-matching training module for residual prediction.
    Trains the network to predict the velocity field (residual - noise)
    from an intermediate interpolated state.
    """
    r'''
    Base class for training modules. All training modules must
    inherit from this class and implement the required methods.
    This class handles the training logic of the surrogate model.
    To implement a specific training procedure, inherit from this class
    and implement the `estimate_loss` method, which is the main method
    to define the training logic. The additional `estimate_auxiliary_losses`
    method can be overridden to add extra loss terms if needed during
    validation. While this class provides default implementations for
    training, validation, and optimizer configuration, these can also be
    overridden if custom behavior is required. Logic for testing and prediction
    is not implemented in this class, as these are expected to be handled
    by the forecast modules using the trained surrogate model.
    '''
    def __init__(
            self,
            network: torch.nn.Module,
            pre_pipeline: PrePipeline,
            post_pipeline: PostPipeline,
            lr: float = 3e-4,
            lr_warmup_steps: int = 5000,
            total_steps: int = 100000,
            weight_decay: float = 0.0,
            ema_rate: float = 0.9999,
            epsilon: float = 1e-8,
    ):
        r'''
        Initialize the training module.

        Parameters
        ----------
        network : torch.nn.Module
            The neural network model to be trained.
        pre_pipeline : PrePipeline
            The sequence of pre-processing modules to apply to the input data.
        post_pipeline : PostPipeline
            The sequence of post-processing modules to apply to the output data.
        lr : float, optional
            Learning rate for the optimizer, by default 3e-4.
        lr_warmup_steps : int, optional
            Number of warmup steps for the learning rate scheduler,
            by default 5000.
        total_steps : int, optional
            Total number of training steps, by default 100000.
        weight_decay : float, optional
            Weight decay for the optimizer, by default 0.0.
        ema_rate : float, optional
            Decay rate for EMA, by default 0.9999.
        epsilon : float, optional
            Epsilon value for numerical stability in optimizers,
            by default 1e-8.
        '''
        super().__init__()
        # Network definitions
        self.network = network
        self.pre_pipeline = pre_pipeline
        self.post_pipeline = post_pipeline

        # Exponential Moving Average (EMA) parameters
        self.ema_rate = ema_rate
        self.ema_network = torch.optim.swa_utils.AveragedModel(
            self.network,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.ema_rate
            ),
            device="cpu"
        )
        self.ema_network.requires_grad_(False)

        # Learning rate parameters
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.total_steps = total_steps
        self.weight_decay = weight_decay

        self.epsilon = epsilon

    def state_dict(
            self,
            *args: object,
            **kwargs: object,
    ) -> dict:
        r'''
        Return the state dictionary with compiled-module
        prefixes removed.

        When ``torch.compile`` wraps ``self.network``,
        state-dict keys acquire an ``_orig_mod.`` infix
        (e.g. ``network._orig_mod.linear.weight``).  This
        override strips that infix so that checkpoints are
        always saved with clean ``network.*`` keys,
        ensuring compatibility with forecast-time loading.

        Returns
        -------
        dict
            State dictionary with ``._orig_mod`` removed
            from all keys.
        '''
        raw = super().state_dict(*args, **kwargs)
        return {
            key.replace("._orig_mod", ""): value
            for key, value in raw.items()
        }

    def on_train_batch_end(self, *args, **kwargs) -> None:
        r'''
        Callback method called at the end of each training batch. This method
        updates the EMA network parameters if EMA is activated.
        '''
        self.ema_network.update_parameters(self.network)

    def on_train_end(self) -> None:
        r'''
        Callback method called at the end of training. This method updates the
        batch normalization statistics of the EMA network if EMA is activated.
        '''
        torch.optim.swa_utils.update_bn(
            self.trainer.train_dataloader, self.ema_network
        )

    def training_step(
            self,
            batch: Dict[str, Any],
            batch_idx: int
    ) -> Dict[str, Any]:
        r'''
        Training step for a given batch of data. This method calls the
        `estimate_loss` method to compute the loss.

        Parameters
        ----------
        batch : Dict[str, Any]
            A batch of data containing `states_surface` and `states_levels`
            tensors.
        batch_idx : int
            The index of the current batch. Unused.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the computed `loss` and any additional
            outputs.
        '''
        outputs = self.estimate_loss(
            batch=batch, prefix="train"
        )
        return outputs

    def validation_step(
            self,
            batch: Dict[str, Any],
            batch_idx: int
    ) -> Dict[str, Any]:
        r'''
        Validation step for a given batch of data. This method calls the
        `estimate_loss` method to compute the loss and
        `estimate_auxiliary_losses` method to compute any additional losses.

        Parameters
        ----------
        batch : Dict[str, Any]
            A batch of data containing `states_surface` and `states_levels`
            tensors.
        batch_idx : int
            The index of the current batch. Unused.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the computed `loss` and any additional
            outputs.
        '''
        outputs = self.estimate_loss(
            batch=batch, prefix="val"
        )
        self.estimate_auxiliary_losses(
            batch, outputs, prefix="val"
        )
        return outputs

    def test_step(
            self,
            batch: Dict[str, Any],
            batch_idx: int
    ) -> None:
        r'''
        No testing logic is implemented in the training module, as testing is
        expected to be handled by the forecast modules using the trained
        surrogate model. This method raises a ValueError if called.
        
        Raises
        ------
        ValueError
            Always raised to indicate that testing is not implemented in the
            training module.
        '''
        raise ValueError(
            "Testing not implemented for the surrogate model. "
            "Please perform instead forecasts with the appropriate forecast "
            "module and the trained model, and evaluate these forecasts."
        )

    def predict_step(
            self,
            batch: Dict[str, Any],
            batch_idx: int
    ) -> None:
        r'''
        No prediction logic is implemented in the training module, as
        prediction is expected to be handled by the forecast modules using the
        trained surrogate model. This method raises a ValueError if called.
        
        Raises
        ------
        ValueError
            Always raised to indicate that prediction is not implemented in the
            training module.
        '''
        raise ValueError(
            "Predictions are not implemented for the surrogate model. "
            "Please perform instead forecasts with the appropriate forecast "
            "module and the trained model, and evaluate these forecasts."
        )

    def configure_optimizers(
            self
    ) -> Dict[str, Any]:
        r'''
        Configure the optimizers and learning rate schedulers for training.
        The AdamW optimizer is used with weight decay applied only to certain
        parameters. A cosine annealing learning rate scheduler with a linear
        warmup is used to adjust the learning rate during training.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the optimizer and learning rate scheduler
            configurations.
        '''
        wd_params, nowd_params = split_wd_params(self.network)
        pipeline_params = list(
            self.pre_pipeline.parameters()
        ) + list(
            self.post_pipeline.parameters()
        )
        nowd_params += pipeline_params
        optimizer = torch.optim.AdamW([
            {"params": wd_params, "weight_decay": self.weight_decay},
            {"params": nowd_params, "weight_decay": 0.0}
        ], lr=self.lr, betas=(0.9, 0.99))
        remaining_steps = self.total_steps - self.lr_warmup_steps
        if self.lr_warmup_steps > 0 and remaining_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=1E-3,
                total_iters=self.lr_warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=remaining_steps,
                eta_min=1E-6,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.lr_warmup_steps],
            )
        elif self.lr_warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=1E-3,
                total_iters=self.lr_warmup_steps,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=max(1, self.total_steps),
                eta_min=1E-6,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    @abc.abstractmethod
    def estimate_loss(
            self,
            batch: Dict[str, Any],
            prefix: str = "train"
    ) -> Dict[str, Any]:
        # batch is expected to contain:
        #   "input":    normalized state s_t,   shape (B, 1, H, W)
        #   "residual": normalized s_{t+dt} - s_t, shape (B, 1, H, W)
        batch_in  = batch["input"]
        batch_res = batch["residual"]

        # Sample noise and pseudo-time (flow matching schedule)
        noise = torch.randn_like(batch_res)

        pseudo_time = torch.linspace(
            0, 1, batch_res.shape[0],
            device=batch_res.device, dtype=batch_res.dtype
        )
        time_shift = torch.rand(
            1, device=batch_res.device, dtype=batch_res.dtype
        )
        pseudo_time = (pseudo_time + time_shift) % 1
        pseudo_time_4d = pseudo_time.view(-1, 1, 1, 1)

        # Interpolate between noise and target residual
        intermediate_state = (
            pseudo_time_4d * batch_res
            + (1 - pseudo_time_4d) * noise
        )
        target_velocity = batch_res - noise

        # Network input: concat intermediate state with conditioning
        input_tensor = torch.cat((intermediate_state, batch_in), dim=1)

        # Forward pass through the network (pre/post pipelines handled outside
        # or you can call self.pre_pipeline / self.post_pipeline here)
        prediction = self.network(input_tensor, pseudo_time)

        # MSE loss on velocity field
        loss = (prediction - target_velocity).pow(2).mean()

        self.log(
            f"{prefix}/loss", loss,
            on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True
        )

        return {"loss": loss}

    def estimate_auxiliary_losses(
            self,
            batch: Dict[str, Any],
            outputs: Dict[str, Any],
            prefix: str = "train"
    ) -> None:
        r'''
        Method to estimate auxiliary losses for a given batch of data. This can
        be overridden by subclasses to add additional loss terms.

        Parameters
        ----------
        batch : Dict[str, Any]
            A batch of data containing `state_surface`, `state_levels`, and
            `forcings` tensors.
        outputs : Dict[str, Any]
            The outputs from the `estimate_loss` method for the given batch.
        prefix : str, optional
            A prefix for logging purposes, by default "train".
        '''
        pass
