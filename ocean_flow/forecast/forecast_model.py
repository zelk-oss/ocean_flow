#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

# System modules
import logging
from typing import Dict, Optional, Tuple

# External modules
import lightning.fabric
import torch
import numpy as np

# Internal modules
from ocean_flow.modules.forecast_module import ForecastModule


main_logger = logging.getLogger(__name__)


__all__ = [
    "ForecastModel",
]


class ForecastModel(object):
    r'''
    Stateful autoregressive wrapper around a ForecastModule.

    Manages rolling-window state, auxiliary inputs, and
    per-step forcings. Device placement and mixed-precision
    autocast are delegated entirely to a
    ``lightning.fabric.Fabric`` instance so that single-GPU,
    multi-GPU, and CPU execution are handled transparently
    without explicit device-management boilerplate.

    The intended deployment is on the main process under
    ``lightning.fabric.Fabric``; Dask workers handle IO
    via ``client.submit`` futures while inference stays
    on the main process.

    Parameters
    ----------
    module : ForecastModule
        Module to wrap for autoregressive forecasting.
    fabric : lightning.fabric.Fabric
        Configured Fabric instance used for device placement
        (``fabric.setup``), tensor movement
        (``fabric.to_device``), and mixed-precision context
        (``fabric.autocast``).
    dtype : torch.dtype, optional
        Data type for state tensors. Default is
        ``torch.float32``.
    compile : bool, optional
        If ``True``, compile ``module.network`` with
        ``torch.compile`` before ``fabric.setup``.
        Default is ``False``.
    n_in_steps : int, optional
        Number of input time steps the module expects.
        Default is 1.
    n_out_steps : int, optional
        Number of output time steps per call. Default is 1.

    Attributes
    ----------
    fabric : lightning.fabric.Fabric
        The Fabric instance used for device/precision control.
    dtype : torch.dtype
        Data type used for state and auxiliary tensors.
    n_in_steps : int
        Rolling-window width kept in internal state.
    n_out_steps : int
        Number of prediction steps returned per module call.
    '''

    def __init__(
            self,
            module: ForecastModule,
            fabric: lightning.fabric.Fabric,
            dtype: torch.dtype = torch.float32,
            compile: bool = False,
            n_in_steps: int = 1,
            n_out_steps: int = 1,
    ) -> None:
        r'''
        Initialize the ForecastModel.

        Parameters
        ----------
        module : ForecastModule
            The ForecastModule to wrap.  Moved to the correct
            dtype and then handed to ``fabric.setup``.
        fabric : lightning.fabric.Fabric
            Pre-constructed Fabric instance that handles
            device placement, multi-GPU distribution, and
            mixed-precision context.
        dtype : torch.dtype, optional
            Data type for state tensors and weight casting.
            Default is ``torch.float32``.
        compile : bool, optional
            Whether to compile ``module.network`` via
            ``torch.compile`` (``reduce-overhead`` mode)
            before ``fabric.setup``.  Default is ``False``.
        n_in_steps : int, optional
            Width of the rolling input window.  Must be a
            positive integer.  Default is 1.
        n_out_steps : int, optional
            Number of output time steps the module produces.
            Must be a positive integer.  Default is 1.

        Raises
        ------
        ValueError
            If ``n_in_steps`` is not a positive integer.
        ValueError
            If ``n_out_steps`` is not a positive integer.
        '''
        self._state: Dict[str, torch.Tensor] = None
        self._auxiliary: Optional[
            Dict[str, torch.Tensor]
        ] = None
        self._module = None
        self.fabric = fabric
        self.dtype = dtype
        self.compile = compile

        if not isinstance(n_in_steps, int) or n_in_steps < 1:
            raise ValueError(
                "n_in_steps must be an integer >= 1"
            )
        if (
            not isinstance(n_out_steps, int)
            or n_out_steps < 1
        ):
            raise ValueError(
                "n_out_steps must be an integer >= 1"
            )

        self.n_in_steps = n_in_steps
        self.n_out_steps = n_out_steps
        self.module = module

    @property
    def device(self) -> torch.device:
        r'''
        Primary device as reported by Fabric.

        Returns
        -------
        torch.device
            The device managed by ``self.fabric``.
        '''
        return self.fabric.device

    @property
    def module(self) -> torch.nn.Module:
        r'''
        Underlying PyTorch module used for forecasting.

        Returns
        -------
        torch.nn.Module
            The wrapped forecast module.
        '''
        return self._module

    @module.setter
    def module(self, module: torch.nn.Module) -> None:
        r'''
        Set the module, casting dtype and calling fabric.setup.

        Casts the module weights to ``self.dtype``, optionally
        compiles ``module.network``, then calls
        ``self.fabric.setup(module)`` for device placement and
        optional multi-GPU distribution.

        Parameters
        ----------
        module : ForecastModule
            The ForecastModule to wrap.
        '''
        module = module.to(dtype=self.dtype)
        if self.compile:
            module.network = torch.compile(
                module.network, mode="reduce-overhead"
            )
        self._module = self.fabric.setup(module)

    @property
    def state(self) -> Dict[str, torch.Tensor]:
        r'''
        Current state as a dict of PyTorch tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            The current model state.

        Raises
        ------
        ValueError
            If the state has not been initialized yet.
        '''
        if self._state is None:
            raise ValueError(
                "State has not been initialized yet."
            )
        return self._state

    def _dict_to_tensor(
            self,
            in_dict: Dict[str, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        r'''
        Convert numpy arrays to tensors via fabric.to_device.

        Creates tensors with ``self.dtype`` and moves them to
        the Fabric-managed device via ``fabric.to_device``.

        Parameters
        ----------
        in_dict : Dict[str, np.ndarray]
            Input arrays to convert.

        Returns
        -------
        Dict[str, torch.Tensor]
            Corresponding tensors on the Fabric device with
            ``self.dtype``.
        '''
        return {
            k: self.fabric.to_device(
                torch.as_tensor(v, dtype=self.dtype)
            )
            for k, v in in_dict.items()
        }

    def set_state(
            self, state: Dict[str, np.ndarray],
    ) -> None:
        r'''
        Initialize the model state from numpy arrays.

        Parameters
        ----------
        state : Dict[str, np.ndarray]
            Arrays to initialize the model state from.
        '''
        self._state = self._dict_to_tensor(state)

    def get_state(self) -> Dict[str, np.ndarray]:
        r'''
        Return the current model state as numpy arrays.

        Returns
        -------
        Dict[str, np.ndarray]
            The current state on CPU as NumPy arrays.
        '''
        return {
            k: v.cpu().numpy()
            for k, v in self.state.items()
        }

    def set_auxiliary(
            self,
            auxiliary: Dict[str, np.ndarray],
    ) -> None:
        r'''
        Set auxiliary variables from numpy arrays.

        Auxiliary variables are static in time and are
        forwarded to the module at every step.

        Parameters
        ----------
        auxiliary : Dict[str, np.ndarray]
            Arrays to use as auxiliary variables.
        '''
        self._auxiliary = self._dict_to_tensor(auxiliary)

    def _validate_output_tuple(
            self,
            output_tuple: Tuple[torch.Tensor, ...],
    ) -> None:
        r'''
        Validate module output against configured step counts.

        Parameters
        ----------
        output_tuple : Tuple[torch.Tensor, ...]
            Output tensors returned by the forecast module.

        Raises
        ------
        ValueError
            If ``len(output_tuple)`` does not equal
            ``len(self._state)``.
        ValueError
            If a tensor's dim-1 size does not equal
            ``n_out_steps``.
        '''
        if len(output_tuple) != len(self._state):
            raise ValueError(
                f"Module returned {len(output_tuple)} "
                f"tensor(s), expected {len(self._state)} "
                f"to match state keys: "
                f"{list(self._state.keys())}"
            )
        for tensor in output_tuple:
            if tensor.shape[1] != self.n_out_steps:
                raise ValueError(
                    f"Module output dim 1 is "
                    f"{tensor.shape[1]}, expected "
                    f"n_out_steps={self.n_out_steps}"
                )

    def _update_rolling_state(
            self,
            output_tuple: Tuple[torch.Tensor, ...],
    ) -> None:
        r'''
        Update the rolling-window state with new predictions.

        Concatenates the current state with the module output
        along dimension 1 and keeps only the last
        ``n_in_steps`` time steps.

        Parameters
        ----------
        output_tuple : Tuple[torch.Tensor, ...]
            Module output tensors, one per state variable.
        '''
        new_state = {}
        for i, var_name in enumerate(self._state.keys()):
            old_window = self._state[var_name]
            new_prediction = output_tuple[i]
            combined = torch.cat(
                [old_window, new_prediction], dim=1,
            )
            new_state[var_name] = (
                combined[:, -self.n_in_steps:]
            )
        self._state = new_state

    def _autoregressive_step(
            self,
            forcings: Optional[
                Dict[str, np.ndarray]
            ] = None,
    ) -> Tuple[torch.Tensor, ...]:
        r'''
        Advance the model by one autoregressive step.

        Applies the module to the current state (plus optional
        auxiliary and forcing inputs) inside
        ``fabric.autocast()`` and updates the internal rolling
        window state.

        Parameters
        ----------
        forcings : Dict[str, np.ndarray] or None, optional
            Time-varying forcing variables for this step.
            Default is ``None``.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Module output with each tensor shaped
            ``(batch, n_out_steps, channels, ...)``.

        Raises
        ------
        ValueError
            If any output tensor has shape[1] != n_out_steps.
        '''
        input_dict = self.state.copy()
        if self._auxiliary is not None:
            input_dict.update(self._auxiliary)
        if forcings is not None:
            input_dict.update(
                self._dict_to_tensor(forcings)
            )
        with self.fabric.autocast(), torch.inference_mode():
            output_tuple = self.module(**input_dict)

        if not isinstance(output_tuple, tuple):
            output_tuple = (output_tuple,)

        self._validate_output_tuple(output_tuple)
        self._update_rolling_state(output_tuple)
        return output_tuple

    def _slice_forcing_window(
            self,
            forcings: Dict[str, np.ndarray],
            step_index: int,
    ) -> Dict[str, np.ndarray]:
        r'''
        Extract a forcing window for a single step.

        Slices each forcing array along dimension 1 to
        produce the window of
        ``n_in_steps + n_out_steps`` time steps needed for
        one autoregressive call.

        Parameters
        ----------
        forcings : Dict[str, np.ndarray]
            Full forcing arrays.
        step_index : int
            Current step index.

        Returns
        -------
        Dict[str, np.ndarray]
            Sliced forcing window.
        '''
        start_idx = step_index * self.n_out_steps
        end_idx = (
            start_idx
            + self.n_in_steps
            + self.n_out_steps
        )
        return {
            k: v[:, start_idx:end_idx]
            for k, v in forcings.items()
        }

    def advance(
            self,
            n: int = 1,
            forcings: Optional[
                Dict[str, np.ndarray]
            ] = None,
    ) -> Dict[str, np.ndarray]:
        r'''
        Advance the model by ``n`` autoregressive steps.

        Calls :meth:`_autoregressive_step` ``n`` times, slicing forcings
        along the time dimension at each call. Predictions are
        concatenated into arrays with shape
        ``(batch, n * n_out_steps, ...)``.

        Parameters
        ----------
        n : int, optional
            Number of autoregressive calls. Default is 1.
        forcings : Dict[str, np.ndarray] or None, optional
            Per-step forcings with shape
            ``(batch, n_in_steps + n * n_out_steps, ...)``.
            Pass ``None`` to run without forcings.
            Default is ``None``.

        Returns
        -------
        Dict[str, np.ndarray]
            Predictions concatenated over ``n`` calls with
            shape ``(batch, n * n_out_steps, ...)``.
        '''
        curr_forcings = None
        accum: Dict[str, list] = {k: [] for k in self._state}
        for i in range(n):
            if forcings is not None:
                curr_forcings = (
                    self._slice_forcing_window(
                        forcings, i,
                    )
                )
            output_tuple = self._autoregressive_step(
                forcings=curr_forcings,
            )
            for j, var_name in enumerate(
                self._state.keys()
            ):
                accum[var_name].append(
                    output_tuple[j].detach().cpu()
                )
        return {
            k: torch.cat(accum[k], dim=1).numpy()
            for k in accum
        }
