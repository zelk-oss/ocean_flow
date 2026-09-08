#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

# System modules
import logging
from typing import Any, Dict

# External modules
import torch

# Internal modules
from flow_train.train_module import TrainingModule


main_logger = logging.getLogger(__name__)


class FlowMatchingTrainingModule(TrainingModule):
    r"""
    Flow-matching training module for residual prediction.

    Trains the network to predict the velocity field (residual - noise)
    from an intermediate interpolated state.
    """

    @staticmethod
    def _assert_normalized(
            tensor: torch.Tensor,
            name: str,
            mean_tol: float = 1e-2,
            std_tol: float = 0.25,
    ) -> None:
        """Check that a latent tensor is roughly normalized.

        This assertion is intentionally conservative to allow batch-level
        fluctuations while still catching gross normalization failures.
        """
        assert torch.isfinite(tensor).all(), (
            f"{name} contains non-finite values after preprocessing"
        )
        mean = tensor.mean()
        std = tensor.std()
        assert abs(mean) <= mean_tol, (
            f"{name} mean is not close to 0 after preprocessing: {mean.item():.4e}"
        )
        assert abs(std - 1.0) <= std_tol, (
            f"{name} std is not close to 1 after preprocessing: {std.item():.4e}"
        )

    def estimate_loss(
            self,
            batch: Dict[str, Any],
            prefix: str = "train"
    ) -> Dict[str, Any]:
        # batch is expected to contain:
        #   "input":    physical state s_t,   shape (B, 1, H, W)
        #   "residual": physical s_{t+dt} - s_t, shape (B, 1, H, W)
        state = batch["input"]
        residual = batch["residual"]

        # Pre-process the conditioning state before it enters the network.
        state_latent = self.pre_pipeline(state)

        # Convert the physical target state into latent residual space.
        target_state = state + residual
        target_latent = self.post_pipeline.to_latent(
            target_state,
            state,
        )

        self._assert_normalized(state_latent, "state_latent")
        self._assert_normalized(target_latent, "target_latent")

        # Sample noise and pseudo-time (flow matching schedule)
        noise = torch.randn_like(target_latent)

        pseudo_time = torch.linspace(
            0, 1, target_latent.shape[0],
            device=target_latent.device, dtype=target_latent.dtype
        )
        time_shift = torch.rand(
            1, device=target_latent.device, dtype=target_latent.dtype
        )
        pseudo_time = (pseudo_time + time_shift) % 1
        pseudo_time_4d = pseudo_time.view(-1, 1, 1, 1)

        # Interpolate between noise and target residual in latent space.
        intermediate_state = (
            pseudo_time_4d * target_latent
            + (1 - pseudo_time_4d) * noise
        )
        target_velocity = target_latent - noise

        # Network input: concat intermediate state with latent conditioning.
        input_tensor = torch.cat((intermediate_state, state_latent), dim=1)

        prediction = self.network(input_tensor, pseudo_time)

        # Log normalization statistics for network inputs and outputs.
        self.log(
            f"{prefix}/state_latent_mean", state_latent.mean(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            f"{prefix}/state_latent_std", state_latent.std(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            f"{prefix}/intermediate_state_mean", intermediate_state.mean(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            f"{prefix}/intermediate_state_std", intermediate_state.std(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            f"{prefix}/input_mean", input_tensor.mean(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            f"{prefix}/input_std", input_tensor.std(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            f"{prefix}/prediction_mean", prediction.mean(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            f"{prefix}/prediction_std", prediction.std(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            f"{prefix}/target_velocity_mean", target_velocity.mean(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            f"{prefix}/target_velocity_std", target_velocity.std(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )

        # MSE loss on the latent velocity field.
        loss = (prediction - target_velocity).pow(2).mean()

        self.log(
            f"{prefix}/loss", loss,
            on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True
        )

        return {"loss": loss}
