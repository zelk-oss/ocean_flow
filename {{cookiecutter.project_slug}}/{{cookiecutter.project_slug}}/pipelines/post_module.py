#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules
import abc
import logging

# External modules
import torch

# Internal modules


main_logger = logging.getLogger(__name__)


class PostModule(torch.nn.Module):
    r'''
    Abstract class for post-processing, transforming from latent space
    back into physical space. A simple example can be a denormalization layer,
    multiplying by a pre-computed standard deviation and adding a pre-computed
    mean. Another example can be a layer converting tendencies into absolute
    physical quantities, while clipping the predictions into physical bounds. 
    '''
    @abc.abstractmethod
    def to_latent(
            self,
            target: torch.Tensor,
            initial: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        An abstract method to convert physical targets into latent space. This
        is useful for loss computations in latent space, e.g., when the loss
        should be computed with normalized targets.

        Parameters
        ----------
        target : torch.Tensor
            The target tensor that needs to be transformed into latent space.
        initial : torch.Tensor,
            The initial conditions tensor that can be used for tendency
            estimation.

        Returns
        -------
        latent : torch.Tensor | None
            The transformed target tensor in latent space.
            Default is None.
        '''
        raise NotImplementedError(
            "The to_latent method must be implemented in the "
            "PostProcessing subclass."
        )

    @abc.abstractmethod
    def forward(
            self,
            prediction: torch.Tensor,
            initial: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        Forward pass of the post-processing, transforming from latent space
        back into physical space.

        Parameters
        ----------
        prediction : torch.Tensor
            The prediction tensor containing the output of the forecasting
            model in latent space.
        initial : torch.Tensor
            The initial conditions tensor that can be used for tendency
            estimation.

        Returns
        -------
        transformed : torch.Tensor
            The transformed state tensor in physical space.
        '''
        raise NotImplementedError(
            "The forward method must be implemented in the "
            "PostProcessing subclass."
        )
