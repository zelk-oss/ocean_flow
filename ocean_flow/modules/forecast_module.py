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

    Ensemble spread is produced entirely upstream: ``ForecastModel``
    and ``InputReader`` repeat the same initial condition across
    ``ensemble_size`` batch rows, and each row receives an
    independent noise draw in :py:meth:`sample_residual_with_flow`.
    This module has no ensemble mechanism of its own.

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
    """

    def __init__(
        self,
        network: torch.nn.Module,
        pre_pipeline: PrePipeline,
        post_pipeline: PostPipeline,
        n_int: int = 20,
    ) -> None:
        super().__init__(
            network=network,
            pre_pipeline=pre_pipeline,
            post_pipeline=post_pipeline,
        )

        if n_int <= 0:
            raise ValueError(f"n_int must be positive, got {n_int}.")

        self.n_int = n_int

    def sample_residual_with_flow(
            self,
            condition: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Sample one residual from the learned conditional flow.

        Parameters
        ----------
        condition : torch.Tensor
            Normalized conditioning state, shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Sampled normalized residual, shape (B, C, H, W).

        Raises
        ------
        ValueError
            If ``condition`` does not have 4 dimensions.
        """

        if condition.ndim != 4:
            raise ValueError(
                "Expected condition with shape (B, C, H, W), "
                f"got shape {tuple(condition.shape)}."
            )

        n_samples = condition.shape[0]
        dynamics = torch.randn_like(condition)
        delta_t = 1.0 / self.n_int

        for i in range(self.n_int):
            t = torch.full(
                (n_samples,),
                i / self.n_int,
                device=condition.device,
                dtype=condition.dtype,
            )

            model_input = torch.cat([dynamics, condition], dim=1)
            velocity = self.network(model_input, t)

            dynamics = dynamics + delta_t * velocity

        return dynamics

    def forward(
        self,
        state: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        r"""
        Perform one stochastic forecasting step.

        Parameters
        ----------
        state : torch.Tensor, optional
            Rolling-window state, as delivered by ``ForecastModel``.
            If not provided, the single tensor contained in
            ``kwargs`` is used, supporting a named input variable
            such as ``q``.

            Expected shape either:

                (B, n_in_steps, H, W)

            or:

                (B, n_in_steps, C, H, W)

            Dim 1 is the rolling time-step window (matching
            ``InputReader``/``ForecastModel``'s convention, not an
            ensemble axis). Only the last time step conditions the
            flow. The 4D form is for a channel-less (scalar) state
            variable such as ``q``.

        kwargs : torch.Tensor
            Optional keyword argument containing the single state tensor.

        Returns
        -------
        tuple of torch.Tensor
            Single-element tuple holding the forecasted next state,
            shape (B, 1, H, W) or (B, 1, C, H, W) matching the
            input's channel convention. Wrapped in a tuple because
            ``ForecastModel`` expects one tensor per state variable.

        Raises
        ------
        ValueError
            If ``state`` is not passed and ``kwargs`` does not
            contain exactly one tensor, if both ``state`` and
            ``kwargs`` are given, or if ``state`` does not have 4 or
            5 dimensions.
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
                "Expected state with shape (B, n_in_steps, H, W) or "
                "(B, n_in_steps, C, H, W), got shape "
                f"{tuple(state.shape)}."
            )

        # Only the last time step conditions the flow.
        current_state = state[:, -1]

        # Insert a channel axis for channel-less (scalar) variables.
        added_channel_axis = current_state.ndim == 3
        if added_channel_axis:
            current_state = current_state[:, None]

        # Normalize the conditioning state.
        condition = self.pre_pipeline(current_state)

        # Sample normalized residual.
        residual = self.sample_residual_with_flow(condition)

        # post_pipeline denormalizes the sampled residual and adds
        # it back to the current state.
        next_state = self.post_pipeline(residual, current_state)

        if added_channel_axis:
            next_state = next_state[:, 0]

        # Insert the n_out_steps=1 axis expected by ForecastModel.
        next_state = next_state.unsqueeze(1)

        return (next_state,)