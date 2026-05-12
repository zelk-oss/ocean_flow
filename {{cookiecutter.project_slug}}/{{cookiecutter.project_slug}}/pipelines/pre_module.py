#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules
import logging
import abc

# External modules
import torch

# Internal modules


main_logger = logging.getLogger(__name__)


class PreModule(torch.nn.Module):
    r'''
    Abstract class for pre-processing, i.e., transforming from physical space
    into a latent space. A simple example can be a normalization layer,
    removing a pre-computed mean and scaling by a pre-computed standard
    deviation.

    All pre-processing methods must inherit from this class and implement the
    forward method.
    '''
    @abc.abstractmethod
    def forward(
            self,
            in_tensor: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        Forward pass of pre-processing. The forward method must
        return the same number of tensors as given input.

        Parameters
        ----------
        in_tensor : torch.Tensor
            The input tensor containing the variables in physical space.

        Returns
        -------
        transformed : torch.Tensor
            The transformed state tensor in latent space.
        '''
        raise NotImplementedError(
            "The forward method must be implemented in the "
            "PreModule subclass."
        )
