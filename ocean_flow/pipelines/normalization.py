#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

# System modules
import logging
from typing import Iterable, Tuple

# External modules
import torch

# Internal modules
from .pre_module import PreModule
from .utils import add_dimensions


main_logger = logging.getLogger(__name__)


class PreNormalization(PreModule):
    r'''
    This layer normalizes the input tensor by removing a pre-computed mean
    and scaling by a pre-computed standard deviation.
    '''
    def __init__(
            self,
            mean: Iterable[float],
            std: Iterable[float],
            add_dims: Tuple[int, ...] = (1, 2),
            epsilon: float = 1e-8
    ):
        r'''
        Initialises the PreNormalization layer with the given mean and
        standard deviation.

        Parameters
        ----------
        mean : torch.Tensor
            The mean tensor for normalization.
        std : torch.Tensor
            The standard deviation tensor for normalization.
        add_dims : Tuple[int, ...], optional
            The dimensions to add for broadcasting the mean and standard
            deviation, by default (1, 2).
        epsilon : float, optional
            A small value added to the standard deviation to avoid division
            by zero, by default 1e-8.
        '''
        super().__init__()
        self.mean: torch.Tensor
        self.std: torch.Tensor
        self.epsilon = epsilon
        self.register_buffer('mean', add_dimensions(
            torch.tensor(mean, dtype=torch.float32),
            add_dims
        ))
        self.register_buffer('std', add_dimensions(
            torch.tensor(std, dtype=torch.float32),
            add_dims
        ))

    def forward(
            self,
            in_tensor: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        Forward pass of the normalization layer.

        Parameters
        ----------
        in_tensor : torch.Tensor
            The input tensor containing the variables in physical space.

        Returns
        -------
        normalized : torch.Tensor
            The normalized state tensor in latent space.
        '''
        return (in_tensor - self.mean) / (self.std + self.epsilon)
