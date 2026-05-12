# -*- coding: utf-8 -*-
r'''Tests for forecast/environment.py -- setup_environment
and initialize_client.'''

# System modules
from typing import Any, List, Optional, Union
from unittest.mock import patch

# External modules
import pytest
from omegaconf import OmegaConf

# Internal modules
from {{cookiecutter.project_slug}}.forecast.environment import (
    setup_environment,
    initialize_client,
)


# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def _env_cfg(
        seed: Optional[int] = None,
) -> OmegaConf:
    r'''Build a minimal setup_environment config.

    Parameters
    ----------
    seed : int or None, optional
        Random seed. If None, seed is omitted.
        Default is None.

    Returns
    -------
    OmegaConf
        Minimal configuration for setup_environment.
    '''
    d: dict[str, Any] = {
        "logging_level": "WARNING",
    }
    if seed is not None:
        d["seed"] = seed
    return OmegaConf.create(d)


def _make_client_cfg(
        scheduler: Optional[
            Union[str, List[str]]
        ] = None,
) -> OmegaConf:
    r'''Build minimal client configuration.

    Parameters
    ----------
    scheduler : str, list of str, or None, optional
        Dask scheduler address(es). Default is None.

    Returns
    -------
    OmegaConf
        Minimal configuration for initialize_client.
    '''
    return OmegaConf.create({
        "dask": {
            "scheduler": scheduler,
            "n_workers": 2,
            "dashboard_address": None,
        },
    })


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestEnvironmentFunctional:
    r'''End-to-end tests for environment setup functions.'''

    def test_setup_environment_returns_none(
            self,
    ) -> None:
        r'''setup_environment returns None.'''
        # Arrange
        cfg = _env_cfg()

        # Act
        result = setup_environment(cfg)

        # Assert
        assert result is None

    def test_setup_environment_does_not_create_fabric(
            self,
    ) -> None:
        r'''setup_environment does not instantiate Fabric.'''
        # Arrange
        cfg = _env_cfg()

        # Act
        with patch(
            "lightning.fabric.Fabric.__init__",
        ) as mock_init:
            setup_environment(cfg)

        # Assert
        mock_init.assert_not_called()

    def test_seed_sets(self) -> None:
        r'''seed=42 calls torch.manual_seed with 42.'''
        # Arrange
        cfg = _env_cfg(seed=42)

        # Act & Assert
        with patch("torch.manual_seed") as mock_seed:
            setup_environment(cfg)
            mock_seed.assert_called_once_with(42)

    def test_no_seed(self) -> None:
        r'''Omitting seed skips torch.manual_seed.'''
        # Arrange
        cfg = _env_cfg()

        # Act & Assert
        with patch("torch.manual_seed") as mock_seed:
            setup_environment(cfg)
            mock_seed.assert_not_called()

    def test_setup_environment_per_worker_seed(
            self,
    ) -> None:
        r'''seed=10 with worker_rank=3 seeds with 13.'''
        # Arrange
        cfg = _env_cfg(seed=10)

        # Act
        with patch(
            "torch.manual_seed",
        ) as mock_torch, patch(
            "numpy.random.seed",
        ) as mock_np:
            setup_environment(cfg, worker_rank=3)

        # Assert
        mock_torch.assert_called_once_with(13)
        mock_np.assert_called_once_with(13)

    def test_setup_environment_no_seed_ignores_rank(
            self,
    ) -> None:
        r'''No seed with worker_rank skips manual_seed.'''
        # Arrange
        cfg = _env_cfg()

        # Act
        with patch("torch.manual_seed") as mock_seed:
            setup_environment(cfg, worker_rank=5)

        # Assert
        mock_seed.assert_not_called()


# -----------------------------------------------------------
# Unit tests
# -----------------------------------------------------------

class TestEnvironmentUnittest:
    r'''Isolated unit tests for environment helpers.'''

    def test_scheduler_path(self) -> None:
        r'''Scheduler path uses distributed.Client.'''
        # Arrange
        scheduler = "tcp://127.0.0.1:8786"
        cfg = _make_client_cfg(scheduler=scheduler)

        # Act
        with patch(
            "{{cookiecutter.project_slug}}"
            ".forecast.environment"
            ".distributed.LocalCluster",
        ) as mock_cluster:
            with patch(
                "{{cookiecutter.project_slug}}"
                ".forecast.environment"
                ".distributed.Client",
            ) as mock_client:
                initialize_client(cfg)

        # Assert
        mock_cluster.assert_not_called()
        mock_client.assert_called_once_with(scheduler)

    def test_local_cluster(self) -> None:
        r'''No scheduler creates LocalCluster with threads.'''
        # Arrange
        cfg = _make_client_cfg(scheduler=None)
        fake_cluster = object()

        # Act
        with patch(
            "{{cookiecutter.project_slug}}"
            ".forecast.environment"
            ".distributed.LocalCluster",
            return_value=fake_cluster,
        ) as mock_cluster:
            with patch(
                "{{cookiecutter.project_slug}}"
                ".forecast.environment"
                ".distributed.Client",
            ) as mock_client:
                initialize_client(cfg)

        # Assert
        mock_cluster.assert_called_once_with(
            n_workers=2,
            threads_per_worker=1,
            dashboard_address=None,
            processes=False,
        )
        mock_client.assert_called_once_with(
            fake_cluster,
        )

    def test_initialize_client_with_dp_rank_selects_address(
            self,
    ) -> None:
        r'''List of addresses selects by dp_rank index.'''
        # Arrange
        addresses = [
            "tcp://10.0.0.1:8786",
            "tcp://10.0.0.2:8786",
            "tcp://10.0.0.3:8786",
        ]
        cfg = _make_client_cfg(scheduler=addresses)

        # Act
        with patch(
            "{{cookiecutter.project_slug}}"
            ".forecast.environment"
            ".distributed.LocalCluster",
        ) as mock_cluster:
            with patch(
                "{{cookiecutter.project_slug}}"
                ".forecast.environment"
                ".distributed.Client",
            ) as mock_client:
                initialize_client(cfg, dp_rank=1)

        # Assert
        mock_cluster.assert_not_called()
        mock_client.assert_called_once_with(
            "tcp://10.0.0.2:8786",
        )

    def test_initialize_client_single_address_ignores_rank(
            self,
    ) -> None:
        r'''String address connects all ranks to same.'''
        # Arrange
        addr = "tcp://127.0.0.1:8786"
        cfg = _make_client_cfg(scheduler=addr)

        # Act
        with patch(
            "{{cookiecutter.project_slug}}"
            ".forecast.environment"
            ".distributed.LocalCluster",
        ) as mock_cluster:
            with patch(
                "{{cookiecutter.project_slug}}"
                ".forecast.environment"
                ".distributed.Client",
            ) as mock_client:
                initialize_client(cfg, dp_rank=2)

        # Assert
        mock_cluster.assert_not_called()
        mock_client.assert_called_once_with(addr)

    def test_initialize_client_null_creates_local_threads_only(
            self,
    ) -> None:
        r'''Null scheduler creates LocalCluster processes=False.'''
        # Arrange
        cfg = _make_client_cfg(scheduler=None)
        fake_cluster = object()

        # Act
        with patch(
            "{{cookiecutter.project_slug}}"
            ".forecast.environment"
            ".distributed.LocalCluster",
            return_value=fake_cluster,
        ) as mock_cluster:
            with patch(
                "{{cookiecutter.project_slug}}"
                ".forecast.environment"
                ".distributed.Client",
            ) as mock_client:
                initialize_client(cfg, dp_rank=0)

        # Assert
        mock_cluster.assert_called_once_with(
            n_workers=2,
            threads_per_worker=1,
            dashboard_address=None,
            processes=False,
        )
        mock_client.assert_called_once_with(
            fake_cluster,
        )
        # Verify processes is NOT read from cfg
        assert "processes" not in cfg.dask
