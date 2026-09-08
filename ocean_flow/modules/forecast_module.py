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

        if condition.ndim == 4:
            B, C, H, W = condition.shape
            flat_condition = condition
            dynamics = torch.randn_like(flat_condition)
            n_samples = B

        elif condition.ndim == 5:
            E, B, C, H, W = condition.shape
            flat_condition = condition.reshape(E * B, C, H, W)
            dynamics = torch.randn_like(flat_condition)
            n_samples = E * B

        else:
            raise ValueError(
                "Expected condition with shape (B, C, H, W) or "
                f"(E, B, C, H, W), got shape {tuple(condition.shape)}."
            )

        delta_t = 1.0 / self.n_int

        for i in range(self.n_int):
            t = torch.full(
                (n_samples,),
                i / self.n_int,
                device=condition.device,
                dtype=condition.dtype,
            )

            model_input = torch.cat([dynamics, flat_condition], dim=1)
            velocity = self.network(model_input, t)

            dynamics = dynamics + delta_t * velocity

        if condition.ndim == 4:
            return dynamics.reshape(B, C, H, W)

        return dynamics.reshape(E, B, C, H, W)

    def forward(
        self,
        state: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Perform one stochastic forecasting step.

        Parameters
        ----------
        state : torch.Tensor, optional
            Current physical state. If not provided, the single tensor
            contained in ``kwargs`` is used. This supports passing the
            input variable under a name such as ``q``.

            Expected shape either:

                (B, C, H, W)

            or:

                (E, B, C, H, W)

        kwargs : torch.Tensor
            Optional keyword argument containing the single state tensor.

        Returns
        -------
        torch.Tensor
            Forecasted next state.

            If input has shape (B, C, H, W), output has shape
            (E, B, C, H, W), where E = n_ensemble.

            If input has shape (E, B, C, H, W), output has the same shape.
        """

        if state is None:
            if len(kwargs) != 1:
                raise ValueError(
                    "FlowMatchingForecastModule.forward requires exactly one"
                    " state tensor when state is not passed positionally."
                )
            state = next(iter(kwargs.values()))
        elif kwargs:
            raise ValueError(
                "FlowMatchingForecastModule.forward received both a positional "
                "state and additional keyword arguments."
            )

        if state.ndim not in (4, 5):
            raise ValueError(
                "Expected state with shape (B, C, H, W) or "
                f"(E, B, C, H, W), got shape {tuple(state.shape)}."
            )

        # Normalize the conditioning state.
        condition = self.pre_pipeline(state)

        # Sample normalized residual.
        residual = self.sample_residual_with_flow(condition)

        if state.ndim == 5:
            E, B, C, H, W = state.shape
            state = state.reshape(E * B, C, H, W)
            residual = residual.reshape(E * B, C, H, W)

        # post_pipeline denormalizes the sampled residual and adds it back to state.
        next_state = self.post_pipeline(residual, state)

        if next_state.ndim == 4:
            next_state = next_state.unsqueeze(1)

        return next_state