#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules
import logging

# Internal modules
from .checkpoint import load_forecast_model
from .config import ForecastConfig, generate_forecast_configs
from .environment import setup_environment, initialize_client
from .forecast_model import ForecastModel
from .input import InputReader, dataset_to_numpy_dict
from .output import OutputWriter
from .restart import (
    check_written_regions,
    filter_forecast_configs,
)
from .runner import (
    PrefetchIterator,
    run_forecast,
    run_batch,
    initialize_io,
)
from .validation import (
    validate_restart_config,
    validate_initial_conditions,
    validate_auxiliary,
    validate_forcing,
    validate_checkpoint,
    validate_output_store,
    create_output_store,
    validate_dask_addresses,
)


main_logger = logging.getLogger(__name__)


__all__ = [
    "ForecastConfig",
    "generate_forecast_configs",
    "ForecastModel",
    "InputReader",
    "dataset_to_numpy_dict",
    "OutputWriter",
    "check_written_regions",
    "filter_forecast_configs",
    "run_forecast",
    "PrefetchIterator",
    "run_batch",
    "initialize_io",
    "load_forecast_model",
    "setup_environment",
    "initialize_client",
    "validate_restart_config",
    "validate_initial_conditions",
    "validate_auxiliary",
    "validate_forcing",
    "validate_checkpoint",
    "validate_output_store",
    "create_output_store",
    "validate_dask_addresses",
]
