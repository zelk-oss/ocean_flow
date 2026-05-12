#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules
import logging
from typing import Dict

# External modules
import torch

# Internal modules
from .pre_module import PreModule
from .post_module import PostModule


main_logger = logging.getLogger(__name__)


class PrePipeline(torch.nn.ModuleDict, PreModule):
    r'''
    Pipeline of pre-processing layers. This class allows to sequentially apply
    multiple pre-processing transformations to the input tensor.
    '''
    def __init__(self, **modules: PreModule) -> None:
        super().__init__()
        self.update(modules)

    def forward(
            self,
            in_tensor: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        Forward pass through the chain of pre-processing layers.

        Parameters
        ----------
        in_tensor : torch.Tensor
            The input tensor containing the variables in physical space.

        Returns
        -------
        transformed : torch.Tensor
            The transformed state tensor in latent space after applying all
            pre-processing layers.
        '''
        out_tensor = in_tensor
        for layer in self._modules.values():
            out_tensor = layer(out_tensor, *args, **kwargs)
        return out_tensor


class PostPipeline(torch.nn.ModuleDict, PostModule):
    def __init__(self, **modules: PostModule) -> None:
        super().__init__()
        self.update(modules)
        
    def to_latent(
            self,
            target: torch.Tensor,
            initial: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        Passes the target tensor through the inverted chain of post-processing
        layers to convert it into latent space.

        Parameters
        ----------
        target : torch.Tensor
            The target tensor that needs to be transformed into latent space.
        initial : torch.Tensor,
            The initial conditions tensor that can be used for tendency
            estimation.

        Returns
        -------
        latent : torch.Tensor
            The transformed target tensor in latent space after applying all
            post-processing layers.
        '''
        out_tensor = target
        for layer in reversed(self._modules.values()):
            out_tensor = layer.to_latent(
                out_tensor,
                initial,
                *args,
                **kwargs
            )
        return out_tensor

    def forward(
            self,
            prediction: torch.Tensor,
            initial: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        r'''
        Forward pass through the chain of post-processing layers.

        Parameters
        ----------
        prediction : torch.Tensor
            The input tensor in latent space containing the model predictions.
        initial : torch.Tensor,
            The initial conditions tensor that can be used for tendency
            estimation.

        Returns
        -------
        transformed : torch.Tensor
            The transformed state tensor in physical space after applying all
            post-processing layers.
        '''
        out_tensor = prediction
        for layer in self._modules.values():
            out_tensor = layer(
                out_tensor,
                initial,
                *args,
                **kwargs
            )
        return out_tensor
