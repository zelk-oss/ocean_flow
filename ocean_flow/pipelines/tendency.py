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
from .post_module import PostModule
from .utils import add_dimensions


main_logger = logging.getLogger(__name__)


class TendencyPrediction(PostModule):
    r'''
    This layer denormalizes the predicted tendencies by multiplying by a
    pre-computed standard deviation. The denormalized tendencies are then added
    to the initial conditions. This layer is useful when the model
    predicts tendencies in a normalized space and we want to convert them
    back to physical space. Additionally, it enforces physical bounds on the
    final predictions to ensure they remain within physical limits.
    '''
    def __init__(
            self,
            std: Iterable[float],
            add_dims: Tuple[int, ...] = (1, 2)
    ):
        r'''
        Initialises the TendencyPrediction layer with the given standard
        deviation.

        Parameters
        ----------
        std : Iterable[float]
            The pre-computed standard deviation for denormalization of
            the tendencies.
        add_dims : Tuple[int, ...], optional
            The dimensions to add for broadcasting the standard deviation and
            bounds, by default (1, 2).
        '''
        super().__init__()
        self.std: torch.Tensor
        self.register_buffer(
            'std',
            add_dimensions(torch.as_tensor(std, dtype=torch.float32), add_dims)
        )
        
    def to_latent(
            self,
            target: torch.Tensor,
            initial: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        This method converts physical targets into latent space by computing
        the normalized tendencies. It subtracts the initial conditions from
        the target to obtain the tendencies, then divides by the standard
        deviation to normalize them.

        Parameters
        ----------
        target : torch.Tensor
            The physical target tensor.
        initial : torch.Tensor
            The initial conditions tensor associated with the target.

        Returns
        -------
        latent : torch.Tensor
            The normalized tendency tensor in latent space.
        '''
        tendencies = target - initial
        latent = tendencies / self.std
        return latent
    
    def forward(
            self,
            prediction: torch.Tensor,
            initial: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        Forward pass of the denormalization layer. This method denormalizes
        the predicted tendencies by multiplying by the standard deviation and
        adds them to the initial conditions.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted tendency tensor in latent space.
        initial : torch.Tensor
            The initial conditions tensor associated with the predictions.

        Returns
        -------
        denormalized : torch.Tensor
        '''
        tendencies = prediction * self.std
        denormalized = initial + tendencies
        return denormalized
