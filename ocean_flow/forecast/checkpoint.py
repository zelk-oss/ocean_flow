#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

r'''Checkpoint loading and state-dict preparation for
forecast inference.
'''

# System modules
import logging
from typing import Any, Dict

# External modules
import lightning.fabric
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

# Internal modules
from ocean_flow.forecast.forecast_model import (
    ForecastModel,
)


main_logger = logging.getLogger(__name__)

__all__ = ["load_forecast_model"]


def _strip_compiled_prefix(
        state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    r'''
    Strip ``._orig_mod`` infixes from state-dict keys.

    Checkpoints saved from ``torch.compile``-wrapped
    networks contain an ``_orig_mod`` path segment
    (e.g. ``network._orig_mod.linear.weight``).  This
    function normalizes those keys back to their
    uncompiled form.

    Parameters
    ----------
    state_dict : Dict[str, Any]
        Raw state dictionary, possibly with compiled
        key prefixes.

    Returns
    -------
    Dict[str, Any]
        State dictionary with clean keys.
    '''
    return {
        key.replace("._orig_mod", ""): value
        for key, value in state_dict.items()
    }


def _unwrap_checkpoint_state_dict(
        checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    r'''
    Return the raw state dict from a checkpoint payload.

    Parameters
    ----------
    checkpoint : Dict[str, Any]
        Checkpoint payload returned by ``torch.load``.

    Returns
    -------
    Dict[str, Any]
        Raw state dictionary to use for model weight
        preparation.
    '''
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _extract_ema_state_dict(
        state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    r'''
    Extract EMA network weights and map them to forecast keys.

    Parameters
    ----------
    state_dict : Dict[str, Any]
        Raw checkpoint state dictionary.

    Returns
    -------
    Dict[str, Any]
        State dictionary with EMA network keys remapped to
        ``network.*`` names.

    Raises
    ------
    ValueError
        If no EMA network weights are present in the
        checkpoint.
    '''
    ema_prefix = "ema_network.module."
    ema_state_dict = {
        f"network.{key[len(ema_prefix):]}": value
        for key, value in state_dict.items()
        if key.startswith(ema_prefix)
    }
    if len(ema_state_dict) == 0:
        raise ValueError(
            "EMA weights were requested, but the checkpoint "
            "does not contain EMA weights."
        )
    return ema_state_dict


def _prepare_checkpoint_state_dict(
        checkpoint: Dict[str, Any],
        forecast_module: torch.nn.Module,
        load_ema: bool = False,
) -> Dict[str, Any]:
    r'''
    Prepare checkpoint weights for loading into the forecast
    module.

    Parameters
    ----------
    checkpoint : Dict[str, Any]
        Checkpoint payload returned by ``torch.load``.
    forecast_module : torch.nn.Module
        Instantiated forecast module receiving the weights.
    load_ema : bool, optional
        If ``True``, remap EMA network weights onto
        ``network.*`` keys before filtering.
        Default is ``False``.

    Returns
    -------
    Dict[str, Any]
        Filtered state dictionary containing only keys used
        by the forecast module.
    '''
    state_dict = _strip_compiled_prefix(
        _unwrap_checkpoint_state_dict(checkpoint)
    )
    prepared_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("ema_network.")
    }
    if load_ema:
        prepared_state_dict.update(
            _extract_ema_state_dict(state_dict)
        )

    forecast_keys = set(
        forecast_module.state_dict().keys()
    )
    return {
        key: value
        for key, value in prepared_state_dict.items()
        if key in forecast_keys
    }


def load_forecast_model(
        cfg: DictConfig,
        fabric: lightning.fabric.Fabric,
) -> ForecastModel:
    r'''
    Load the forecast model with trained weights for inference.

    Optionally compiles the model for faster inference and
    delegates device placement and precision management to a
    ``lightning.fabric.Fabric`` instance.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing:

        - forecast_module : DictConfig
            Hydra target config for the forecast module class.
        - ckpt_path : str or None, optional
            Path to a checkpoint file. ``None`` skips loading.
        - load_ema : bool, optional
            If ``True``, remap EMA network weights onto
            network keys. Default is ``False``.
        - compile : bool, optional
            If ``True``, compile the network with
            ``torch.compile``. Default is ``False``.
        - n_in_steps : int, optional
            Number of input time steps. Default is 1.
        - n_out_steps : int, optional
            Number of output time steps per call.
            Default is 1.
    fabric : lightning.fabric.Fabric
        Fabric instance used for device placement and
        autocast.

    Returns
    -------
    ForecastModel
        Forecast model with loaded weights, ready for
        inference.
    '''
    main_logger.info(
        "Instantiating forecast module "
        f"<{cfg.forecast_module._target_}>"
    )
    # Instantiate the forecast module
    forecast_module: torch.nn.Module = instantiate(
        cfg.forecast_module
    )

    if cfg.get("ckpt_path", None) is not None:
        main_logger.info(
            f"Loading checkpoint from {cfg.ckpt_path}"
        )
        # Load the checkpoint state dict to CPU
        checkpoint = torch.load(
            cfg.ckpt_path,
            map_location='cpu',
            weights_only=True,
        )
        state_dict = _prepare_checkpoint_state_dict(
            checkpoint=checkpoint,
            forecast_module=forecast_module,
            load_ema=cfg.get("load_ema", False),
        )
        load_result = forecast_module.load_state_dict(
            state_dict, strict=False
        )
        # Log any missing or unexpected keys for debugging
        if load_result.missing_keys:
            main_logger.warning(
                "Missing keys when loading checkpoint: %s",
                load_result.missing_keys,
            )
        if load_result.unexpected_keys:
            main_logger.warning(
                "Unexpected keys when loading checkpoint: "
                "%s",
                load_result.unexpected_keys,
            )
        main_logger.info("Checkpoint loaded successfully")

    # Set the model to eval mode
    forecast_module.eval()

    # Set the forecast model wrapper
    forecast_model = ForecastModel(
        module=forecast_module,
        fabric=fabric,
        compile=False,
        n_in_steps=cfg.get("n_in_steps", 1),
        n_out_steps=cfg.get("n_out_steps", 1),
    )
    return forecast_model
