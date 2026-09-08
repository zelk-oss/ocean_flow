#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco
r"""Shared test fixtures for the Ocean Flow Matching test suite.

Only fixtures used by ocean_flow's own project code (modules/, networks/)
live here. Fixtures for the vendored framework layer (data loading,
forecast I/O, pipelines) moved with that code to the flow-core/flow-train/
flow-forecast submodules, which carry their own test suites.
"""

# System modules
import logging

# External modules
import lightning.fabric
import pytest
import torch

# Internal modules
from flow_core.pre_processing import PrePipeline
from flow_core.post_processing import PostPipeline


main_logger = logging.getLogger(__name__)


def make_fabric(
        accelerator: str = "cpu",
        devices: int = 1,
        precision: str = "32-true",
) -> lightning.fabric.Fabric:
    r'''Create a Fabric instance for tests.

    Plain module-level factory (not a pytest fixture), callable
    from any test file without fixture injection.

    Parameters
    ----------
    accelerator : str, optional
        Fabric accelerator string, e.g. ``"cpu"`` or
        ``"gpu"``. Default is ``"cpu"``.
    devices : int, optional
        Number of devices to use. Default is ``1``.
    precision : str, optional
        Precision mode string, e.g. ``"32-true"`` or
        ``"bf16-mixed"``. Default is ``"32-true"``.

    Returns
    -------
    lightning.fabric.Fabric
        Configured ``Fabric`` instance.
    '''
    return lightning.fabric.Fabric(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
    )


class IdentityPreModule(torch.nn.Module):
    r'''
    Identity pre-processing module.

    Passes data through unchanged for testing purposes.
    '''
    def forward(
            self,
            in_tensor: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''
        Return input tensor unchanged.

        Parameters
        ----------
        in_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The same input tensor, unchanged.
        '''
        return in_tensor


class IdentityPostModule(torch.nn.Module):
    r'''
    Identity post-processing module.

    Passes predictions through unchanged for testing.
    '''
    def forward(
            self,
            prediction: torch.Tensor,
            initial: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''
        Return prediction unchanged.

        Parameters
        ----------
        prediction : torch.Tensor
            Predicted tensor.
        initial : torch.Tensor
            Initial state tensor (unused).

        Returns
        -------
        torch.Tensor
            The prediction tensor, unchanged.
        '''
        return prediction

    def to_latent(
            self,
            target: torch.Tensor,
            initial: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''
        Return target unchanged.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor.
        initial : torch.Tensor
            Initial state tensor (unused).

        Returns
        -------
        torch.Tensor
            The target tensor, unchanged.
        '''
        return target


@pytest.fixture()
def pre_pipeline() -> PrePipeline:
    r'''
    Return an identity pre-processing pipeline.

    Returns
    -------
    PrePipeline
        Pipeline with identity modules for surface and
        level variables.
    '''
    return PrePipeline(
        states_surface=IdentityPreModule(),
        states_levels=IdentityPreModule(),
    )


@pytest.fixture()
def post_pipeline() -> PostPipeline:
    r'''
    Return an identity post-processing pipeline.

    Returns
    -------
    PostPipeline
        Pipeline with identity modules for surface and
        level variables.
    '''
    return PostPipeline(
        states_surface=IdentityPostModule(),
        states_levels=IdentityPostModule(),
    )


class DummyNetwork(torch.nn.Module):
    r'''
    Dummy network that outputs zeros with correct shape.

    The output shape matches the test zarr store dimensions
    for surface and level variables.

    Attributes
    ----------
    n_surf_vars : int
        Number of surface variables.
    n_lev_vars : int
        Number of level variables.
    n_levels : int
        Number of vertical levels.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.n_surf_vars = 2
        self.n_lev_vars = 2
        self.n_levels = 3
        # A parameter so the module is not empty
        self.linear = torch.nn.Linear(1, 1)

    def forward(
            self,
            x: torch.Tensor,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''
        Return zeros with the expected output shape.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape
            ``(batch, channels, lat, lon)``.

        Returns
        -------
        torch.Tensor
            Zero tensor with shape
            ``(batch, out_channels, lat, lon)``.
        '''
        batch = x.size(0)
        n_lat, n_lon = x.shape[-2], x.shape[-1]
        out_channels = (
            self.n_surf_vars
            + self.n_lev_vars * self.n_levels
        )
        return torch.zeros(
            batch, out_channels, n_lat, n_lon,
        )


@pytest.fixture()
def dummy_network() -> DummyNetwork:
    r'''
    Return an instance of the DummyNetwork.

    Returns
    -------
    DummyNetwork
        A dummy network for testing.
    '''
    return DummyNetwork()
