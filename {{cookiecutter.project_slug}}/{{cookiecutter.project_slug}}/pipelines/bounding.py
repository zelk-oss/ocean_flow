#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules
import logging
from typing import Tuple, Iterable

# External modules
import torch

# Internal modules
from .post_module import PostModule
from .utils import add_dimensions


main_logger = logging.getLogger(__name__)


class LowerBoundPrediction(PostModule):
    r'''
    This layer enforces a lower bound on the model predictions. It ensures
    that all predicted values are above a specified minimum threshold.
    '''
    def __init__(
            self,
            lower_bound: Iterable[float],
            add_dims: Tuple[int, ...] = (1, 2)
    ):
        r'''
        Initialises the LowerBoundPrediction layer with the given lower
        bound.

        Parameters
        ----------
        lower_bound : Iterable[float]
            The minimum threshold tensor for enforcing lower bounds on the
            predictions.
        '''
        super().__init__()
        self.lower_bound: torch.Tensor
        self.register_buffer(
            'lower_bound',
            add_dimensions(
                torch.as_tensor(lower_bound, dtype=torch.float32), add_dims
            )
        )

    def to_latent(
            self,
            target: torch.Tensor,
            initial: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        Passes the target tensor through.

        Parameters
        ----------
        target : torch.Tensor
            The target tensor in physical space, assumed to be already bounded.
        initial : torch.Tensor,
            The initial conditions tensor (not used in this method).

        Returns
        -------
        latent : torch.Tensor
            The target tensor passed through without modification.
        '''
        return target

    def forward(
            self,
            prediction: torch.Tensor,
            initial: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        Forward pass of the lower bound prediction layer. This method enforces
        the lower bound on the predictions by applying an element-wise
        maximum operation between the predictions and the lower bound.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted tensor from the model in physical space.
        initial : torch.Tensor,
            The initial conditions tensor (not used in this method).

        Returns
        -------
        bounded : torch.Tensor
            The prediction tensor with enforced lower bounds.
        '''
        return prediction.clamp(min=self.lower_bound)
