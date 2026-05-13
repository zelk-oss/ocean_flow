#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

# System modules
import abc
import logging

# External modules
import lightning.pytorch as pl
import torch

# Internal modules
from ocean_flow.pipelines import PrePipeline, PostPipeline


main_logger = logging.getLogger(__name__)


class ForecastModule(pl.LightningModule):
    r"""
    Abstract base class for forecast modules using PyTorch Lightning.

    All forecast modules must inherit from this class and implement the
    :py:meth:`forward` method to define a single prediction step.  The
    base class stores the network and pre/post pipelines; it deliberately
    contains **no** prediction-loop, zarr I/O, or autoregressive-rollout
    logic.

    For running a full prediction pipeline (autoregressive rollout
    and zarr output), use ``ForecastInference`` or
    ``scripts/forecast.py``.

    Parameters
    ----------
    network : torch.nn.Module
        The neural network used for forecasting.
    pre_pipeline : PrePipeline
        The input transformation modules.
    post_pipeline : PostPipeline
        The output transformation modules.

    Attributes
    ----------
    network : torch.nn.Module
        The neural network used for forecasting.
    pre_pipeline : PrePipeline
        The input transformation modules.
    post_pipeline : PostPipeline
        The output transformation modules.
    """

    def __init__(
            self,
            network: torch.nn.Module,
            pre_pipeline: PrePipeline,
            post_pipeline: PostPipeline,
    ) -> None:
        super().__init__()
        self.network = network
        self.pre_pipeline = pre_pipeline
        self.post_pipeline = post_pipeline

    @abc.abstractmethod
    def forward(self, *args: object, **kwargs: object) -> object:
        r"""
        Forward pass performing a single forecast step.

        Subclasses must override this method to implement their
        specific prediction logic.  The signature is intentionally
        open so that subclasses can accept any combination of
        positional and keyword tensors.

        Raises
        ------
        NotImplementedError
            Always raised when called directly on the base class.
        """
        raise NotImplementedError(
            "The forward method must be implemented in the "
            "ForecastModule subclass."
        )


class FlowMatchingForecastModule(ForecastModule):
    r"""
    Forecast module for residual-based conditional flow matching.

    Given a current state x_k, this module performs one stochastic
    forecasting step:

        x_{k+1} = x_k + r,

    where r is sampled by integrating the learned flow model from
    Gaussian noise to the residual distribution conditional on x_k.

    The neural network is expected to take as input

        concat([current_residual_sample, normalized_condition], dim=1)

    and a pseudo-time tensor t, and to return the velocity field.

    Parameters
    ----------
    network : torch.nn.Module
        Flow-matching neural network.
    pre_pipeline : PrePipeline
        Pipeline applied to the current state before conditioning the flow.
        This should contain the input normalization, equivalent to
        (state - in_mean) / in_std in the old script.
    post_pipeline : PostPipeline
        Pipeline applied to the sampled residual. This should contain the
        residual denormalization, equivalent to
        increment * res_std + res_mean in the old script.
    n_int : int
        Number of Euler integration steps used to sample the flow.
    n_ensemble : int
        Number of stochastic ensemble members to generate when the input
        has no ensemble dimension.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        pre_pipeline: PrePipeline,
        post_pipeline: PostPipeline,
        n_int: int = 20,
        n_ensemble: int = 1,
    ) -> None:
        super().__init__(
            network=network,
            pre_pipeline=pre_pipeline,
            post_pipeline=post_pipeline,
        )

        if n_int <= 0:
            raise ValueError(f"n_int must be positive, got {n_int}.")

        if n_ensemble <= 0:
            raise ValueError(f"n_ensemble must be positive, got {n_ensemble}.")

        self.n_int = n_int
        self.n_ensemble = n_ensemble

    def sample_residual_with_flow(self, condition: torch.Tensor) -> torch.Tensor:
        r"""
        Sample one residual from the learned conditional flow.

        Parameters
        ----------
        condition : torch.Tensor
            Normalized conditioning state.

            Expected shape either:

                (B, C, H, W)

            or:

                (E, B, C, H, W)

        Returns
        -------
        torch.Tensor
            Sampled normalized residual with the same shape as ``condition``.
        """

        input_has_ensemble_dim = condition.ndim == 5

        if condition.ndim == 4:
            # Input has shape (B, C, H, W).
            # Add stochastic ensemble dimension.
            condition = condition.unsqueeze(0).repeat(
                self.n_ensemble, 1, 1, 1, 1
            )

        elif condition.ndim != 5:
            raise ValueError(
                "Expected condition with shape (B, C, H, W) or "
                f"(E, B, C, H, W), got shape {tuple(condition.shape)}."
            )

        E, B, C, H, W = condition.shape

        flat_condition = condition.reshape(E * B, C, H, W)

        # Initial sample in residual space: z_0 ~ N(0, I).
        dynamics = torch.randn_like(flat_condition)

        delta_t = 1.0 / self.n_int

        for i in range(self.n_int):
            t = torch.full(
                (E * B,),
                i / self.n_int,
                device=condition.device,
                dtype=condition.dtype,
            )

            model_input = torch.cat([dynamics, flat_condition], dim=1)
            velocity = self.network(model_input, t)

            dynamics = dynamics + delta_t * velocity

        residual = dynamics.reshape(E, B, C, H, W)

        if not input_has_ensemble_dim:
            # Keep ensemble dimension, because the forecast is stochastic.
            return residual

        return residual

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        r"""
        Perform one stochastic forecasting step.

        Parameters
        ----------
        state : torch.Tensor
            Current physical state.

            Expected shape either:

                (B, C, H, W)

            or:

                (E, B, C, H, W)

        Returns
        -------
        torch.Tensor
            Forecasted next state.

            If input has shape (B, C, H, W), output has shape
            (E, B, C, H, W), where E = n_ensemble.

            If input has shape (E, B, C, H, W), output has the same shape.
        """

        input_has_ensemble_dim = state.ndim == 5

        # Normalize the conditioning state.
        # This replaces:
        #     norm_state = (state - in_mean) / in_std
        condition = self.pre_pipeline(state)

        # Sample normalized residual.
        residual = self.sample_residual_with_flow(condition)

        # Denormalize residual.
        # This replaces:
        #     increment_physical = increment * res_std + res_mean
        residual_physical = self.post_pipeline(residual)

        if state.ndim == 4:
            # state: (B, C, H, W)
            # residual_physical: (E, B, C, H, W)
            state = state.unsqueeze(0).repeat(
                self.n_ensemble, 1, 1, 1, 1
            )

        elif state.ndim != 5:
            raise ValueError(
                "Expected state with shape (B, C, H, W) or "
                f"(E, B, C, H, W), got shape {tuple(state.shape)}."
            )

        next_state = state + residual_physical

        return next_state