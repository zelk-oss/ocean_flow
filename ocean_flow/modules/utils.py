#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

# System modules
import logging
from typing import List, Tuple

# External modules
import torch

# Internal modules
from ocean_flow.pipelines import PostPipeline
from ocean_flow.pipelines.pipelines import PrePipeline


main_logger = logging.getLogger(__name__)


def preprocess_data(
        states_surface: torch.Tensor,
        states_levels: torch.Tensor,
        pre_pipeline: PrePipeline,
        post_pipeline: PostPipeline,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_surface, target_surface = states_surface.split(
        (states_surface.size(1)-1, 1), dim=1
    )
    input_levels, target_levels = states_levels.split(
        (states_levels.size(1)-1, 1), dim=1
    )

    # Construct input tensor
    input_tensor = process_inputs(
        states_surface=input_surface,
        states_levels=input_levels,
        pre_pipeline=pre_pipeline,
    )

    # Process targets into tendency space
    latent_surface = post_pipeline["states_surface"].to_latent(
        target_surface.squeeze(1), initial=input_surface[:, -1]
    )
    latent_levels = post_pipeline["states_levels"].to_latent(
        target_levels.squeeze(1), initial=input_levels[:, -1]
    )
    return input_tensor, latent_surface, latent_levels


def process_inputs(
        states_surface: torch.Tensor,
        states_levels: torch.Tensor,
        pre_pipeline: PrePipeline,
) -> torch.Tensor:
    r'''
    Processes the input tensors through the pre-processing pipeline.

    Parameters
    ----------
    states_surface : torch.Tensor
        The input surface state tensor with shape (batch size, time steps,
        number of variables, ...).
    states_levels : torch.Tensor
        The input levels state tensor with shape (batch size, time steps,
        number of variables, ...).
    pre_pipeline : PrePipeline
        The pre-processing pipeline to apply to the input data.

    Returns
    -------
    input_tensor : torch.Tensor
        The processed input tensor after applying the pre-processing pipeline.
    '''
    processed_surface = pre_pipeline["states_surface"](states_surface)
    processed_surface = processed_surface.reshape(
        processed_surface.size(0), -1, *processed_surface.shape[-2:]
    )
    processed_levels = pre_pipeline["states_levels"](states_levels)
    processed_levels = processed_levels.reshape(
        processed_levels.size(0), -1, *processed_levels.shape[-2:]
    )
    input_tensor = torch.cat(
        [processed_surface, processed_levels], dim=1
    )
    return input_tensor


def split_wd_params(
        model: torch.nn.Module
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    # From minGPT https://github.com/karpathy/minGPT
    # Explanation: https://github.com/karpathy/minGPT/pull/24
    decay_params = set()
    no_decay_params = set()
    no_grad_params = set()
    for name, param in model.named_parameters():
        parent_module = model.get_submodule(".".join(name.split(".")[:-1]))
        decay = (
            name.endswith('weight')
            and not isinstance(parent_module, torch.nn.GroupNorm)
            and "norm" not in name
            and "mod" not in name
            and "embedding" not in name
            and "embedder" not in name
            and "log_scale" not in name
            and "ema" not in name
            and "qk_scaling" not in name
        )
        if decay and param.requires_grad:
            decay_params.add(name)
        elif param.requires_grad:
            no_decay_params.add(name)
        else:
            no_grad_params.add(name)

    # Check if all parameters are considered
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay_params & no_decay_params & no_grad_params
    union_params = decay_params | no_decay_params | no_grad_params
    missing_keys = param_dict.keys() - union_params
    if len(inter_params) != 0:
        raise AssertionError(
            "Parameters {0:s} made it into different sets!".format(
                str(inter_params)
            )
        )
    if len(missing_keys) != 0:
        raise AssertionError(
            "Parameters {0:s} were not separated into sets!".format(
                missing_keys
            )
        )

    # Convert into lists of parameters
    decay_params = [param_dict[pn] for pn in sorted(list(decay_params))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay_params))]
    return decay_params, no_decay_params
