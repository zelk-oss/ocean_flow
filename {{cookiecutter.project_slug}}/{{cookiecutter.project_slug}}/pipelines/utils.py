#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules
import logging
from typing import Tuple

# External modules
import torch

# Internal modules


main_logger = logging.getLogger(__name__)


def add_dimensions(
        tensor: torch.Tensor,
        dims: Tuple[int, ...]
) -> torch.Tensor:
    r'''
    Adds singleton dimensions to a tensor at the specified positions.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to which singleton dimensions will be added.
    dims : Tuple[int, ...]
        The positions where singleton dimensions should be added.

    Returns
    -------
    tensor_with_dims : torch.Tensor
        The tensor with added singleton dimensions.
    '''
    for dim in sorted(dims):
        tensor = tensor.unsqueeze(dim)
    return tensor
