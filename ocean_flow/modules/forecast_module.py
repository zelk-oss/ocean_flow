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
