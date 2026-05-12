# -*- coding: utf-8 -*-
r'''Tests for forecast/checkpoint.py -- checkpoint loading,
state dict unwrapping, EMA extraction, and model preparation.

TDD: implementation pending -- checkpoint.py does not exist
yet.  These tests define the contract that the implementation
must satisfy.
'''

# System modules
import logging
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

# External modules
import lightning.fabric
import pytest
import torch
from omegaconf import OmegaConf

# Internal modules
from {{cookiecutter.project_slug}}.modules.forecast_module import (
    ForecastModule,
)
from {{cookiecutter.project_slug}}.forecast.forecast_model import (
    ForecastModel,
)
from {{cookiecutter.project_slug}}.pipelines import (
    PrePipeline,
    PostPipeline,
)
from tests.conftest import (
    IdentityPreModule,
    IdentityPostModule,
    DummyNetwork,
    make_fabric,
)

from {{cookiecutter.project_slug}}.forecast.checkpoint import (
    load_forecast_model,
    _unwrap_checkpoint_state_dict,
    _extract_ema_state_dict,
    _prepare_checkpoint_state_dict,
)


# -----------------------------------------------------------
# Helper classes
# -----------------------------------------------------------

class SimpleForecast(ForecastModule):
    r'''Minimal ForecastModule for checkpoint tests.'''

    def forward(
            self,
            states_surface: torch.Tensor,
            states_levels: torch.Tensor,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r'''Return doubled last time-step of each tensor.'''
        return (
            states_surface[:, -1:] * 2.0,
            states_levels[:, -1:] * 2.0,
        )


# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

_make_test_fabric = make_fabric


def _model_cfg(
        ckpt_path: Optional[str] = None,
        compile_model: bool = False,
        load_ema: bool = False,
) -> OmegaConf:
    r'''Build a minimal load_forecast_model config.

    Parameters
    ----------
    ckpt_path : str or None, optional
        Path to a checkpoint file. Default is None.
    compile_model : bool, optional
        Whether to compile the model. Default is False.
    load_ema : bool, optional
        Whether to load EMA weights. Default is False.

    Returns
    -------
    OmegaConf
        Minimal config for load_forecast_model.
    '''
    return OmegaConf.create({
        "forecast_module": {
            "_target_": (
                "tests.test_forecast_checkpoint"
                ".SimpleForecast"
            ),
        },
        "ckpt_path": ckpt_path,
        "compile": compile_model,
        "load_ema": load_ema,
    })


def _build_simple_forecast() -> SimpleForecast:
    r'''Return a plain SimpleForecast for mocking.

    Returns
    -------
    SimpleForecast
        Forecast module with identity pipelines.
    '''
    return SimpleForecast(
        network=DummyNetwork(),
        pre_pipeline=PrePipeline(
            states_surface=IdentityPreModule(),
            states_levels=IdentityPreModule(),
        ),
        post_pipeline=PostPipeline(
            states_surface=IdentityPostModule(),
            states_levels=IdentityPostModule(),
        ),
    )


def _make_training_checkpoint(
        instance: SimpleForecast,
        network_weight: float = 1.0,
        network_bias: float = 0.0,
        ema_weight: Optional[float] = None,
        ema_bias: Optional[float] = None,
        include_state_dict: bool = True,
) -> Dict[str, Any]:
    r'''Return a training-style checkpoint.

    Parameters
    ----------
    instance : SimpleForecast
        Module whose state_dict provides the key layout.
    network_weight : float, optional
        Fill value for network weight. Default is 1.0.
    network_bias : float, optional
        Fill value for network bias. Default is 0.0.
    ema_weight : float or None, optional
        Fill value for EMA weight. Default is None.
    ema_bias : float or None, optional
        Fill value for EMA bias. Default is None.
    include_state_dict : bool, optional
        Wrap in ``{"state_dict": ...}`` dict. Default True.

    Returns
    -------
    Dict[str, Any]
        Training-style checkpoint dictionary.
    '''
    state_dict = {
        key: value.clone()
        for key, value in instance.state_dict().items()
    }
    state_dict["network.linear.weight"] = torch.full_like(
        state_dict["network.linear.weight"],
        fill_value=network_weight,
    )
    state_dict["network.linear.bias"] = torch.full_like(
        state_dict["network.linear.bias"],
        fill_value=network_bias,
    )
    if ema_weight is not None:
        state_dict[
            "ema_network.module.linear.weight"
        ] = torch.full_like(
            state_dict["network.linear.weight"],
            fill_value=ema_weight,
        )
        state_dict[
            "ema_network.module.linear.bias"
        ] = torch.full_like(
            state_dict["network.linear.bias"],
            fill_value=(
                0.0 if ema_bias is None else ema_bias
            ),
        )
        state_dict[
            "ema_network.n_averaged"
        ] = torch.tensor(8)
    if include_state_dict:
        return {"state_dict": state_dict}
    return state_dict


# Patch target for functions that live in checkpoint.py
_CKPT = (
    "{{cookiecutter.project_slug}}"
    ".forecast.checkpoint"
)


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestCheckpointFunctional:
    r'''End-to-end tests for checkpoint loading functions.'''

    def test_returns_model(self) -> None:
        r'''load_forecast_model returns a ForecastModel.'''
        # Arrange
        cfg = _model_cfg()
        instance = _build_simple_forecast()

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            result = load_forecast_model(
                cfg, fabric=_make_test_fabric(),
            )

        # Assert
        assert isinstance(result, ForecastModel)

    def test_loads_checkpoint(self) -> None:
        r'''torch.load called when ckpt_path provided.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
        )
        instance = _build_simple_forecast()
        fake_state_dict = instance.state_dict()

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=fake_state_dict,
            ) as mock_load:
                load_forecast_model(
                    cfg,
                    fabric=_make_test_fabric(),
                )

                # Assert
                mock_load.assert_called_once()

    def test_lightning_format(self) -> None:
        r'''Lightning checkpoint state_dict key unwrapped.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
        )
        instance = _build_simple_forecast()
        lightning_ckpt = {
            "state_dict": instance.state_dict(),
        }

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=lightning_ckpt,
            ):
                model = load_forecast_model(
                    cfg,
                    fabric=_make_test_fabric(),
                )

        # Assert
        assert isinstance(model, ForecastModel)

    def test_filters_training_only_keys(
            self,
            caplog: pytest.LogCaptureFixture,
    ) -> None:
        r'''Training-only checkpoint keys are filtered out.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
        )
        instance = _build_simple_forecast()
        checkpoint = _make_training_checkpoint(
            instance,
            network_weight=1.5,
            network_bias=2.5,
            ema_weight=9.0,
            ema_bias=10.0,
        )

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=checkpoint,
            ):
                with caplog.at_level(logging.WARNING):
                    model = load_forecast_model(
                        cfg,
                        fabric=_make_test_fabric(),
                    )

        # Assert
        assert isinstance(model, ForecastModel)
        assert len(caplog.records) == 0
        torch.testing.assert_close(
            model.module.network.linear.weight,
            torch.full_like(
                model.module.network.linear.weight,
                fill_value=1.5,
            ),
        )
        torch.testing.assert_close(
            model.module.network.linear.bias,
            torch.full_like(
                model.module.network.linear.bias,
                fill_value=2.5,
            ),
        )

    def test_load_ema_uses_ema_network_weights(
            self,
    ) -> None:
        r'''load_ema selects EMA network weights.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
            load_ema=True,
        )
        instance = _build_simple_forecast()
        checkpoint = _make_training_checkpoint(
            instance,
            network_weight=-1.0,
            network_bias=-2.0,
            ema_weight=7.0,
            ema_bias=8.0,
        )

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=checkpoint,
            ):
                model = load_forecast_model(
                    cfg,
                    fabric=_make_test_fabric(),
                )

        # Assert
        torch.testing.assert_close(
            model.module.network.linear.weight,
            torch.full_like(
                model.module.network.linear.weight,
                fill_value=7.0,
            ),
        )
        torch.testing.assert_close(
            model.module.network.linear.bias,
            torch.full_like(
                model.module.network.linear.bias,
                fill_value=8.0,
            ),
        )

    def test_load_forecast_model_passes_n_in_n_out(
            self,
    ) -> None:
        r'''n_in_steps and n_out_steps forwarded to model.'''
        # Arrange
        cfg = _model_cfg()
        cfg.n_in_steps = 2
        cfg.n_out_steps = 3
        instance = _build_simple_forecast()

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                f"{_CKPT}.ForecastModel",
                autospec=True,
            ) as mock_model:
                load_forecast_model(
                    cfg,
                    fabric=_make_test_fabric(),
                )

                # Assert
                mock_model.assert_called_once()
                call_kwargs = (
                    mock_model.call_args.kwargs
                )
                assert call_kwargs["n_in_steps"] == 2
                assert call_kwargs["n_out_steps"] == 3
                assert isinstance(
                    call_kwargs["fabric"],
                    lightning.fabric.Fabric,
                )

    def test_load_forecast_model_accepts_fabric(
            self,
    ) -> None:
        r'''load_forecast_model stores fabric on model.'''
        # Arrange
        cfg = _model_cfg()
        instance = _build_simple_forecast()
        fabric = _make_test_fabric()

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            model = load_forecast_model(
                cfg, fabric=fabric,
            )

        # Assert
        assert model.fabric is fabric


# -----------------------------------------------------------
# Unit tests
# -----------------------------------------------------------

class TestCheckpointUnittest:
    r'''Isolated unit tests for checkpoint helper functions.'''

    def test_unwrap_with_state_dict_key(self) -> None:
        r'''Returns inner dict when state_dict key present.'''
        # Arrange
        inner = {"network.weight": torch.tensor(1.0)}
        checkpoint = {"state_dict": inner, "epoch": 5}

        # Act
        result = _unwrap_checkpoint_state_dict(checkpoint)

        # Assert
        assert result is inner

    def test_unwrap_without_state_dict_key(
            self,
    ) -> None:
        r'''Returns checkpoint directly when no state_dict.'''
        # Arrange
        checkpoint = {
            "network.weight": torch.tensor(1.0),
        }

        # Act
        result = _unwrap_checkpoint_state_dict(checkpoint)

        # Assert
        assert result is checkpoint

    def test_extract_ema_remaps_keys(self) -> None:
        r'''EMA keys remapped to network.* names.'''
        # Arrange
        state_dict = {
            "ema_network.module.linear.weight": (
                torch.tensor([1.0])
            ),
            "ema_network.module.linear.bias": (
                torch.tensor([2.0])
            ),
            "ema_network.n_averaged": torch.tensor(5),
            "network.linear.weight": (
                torch.tensor([0.0])
            ),
        }

        # Act
        result = _extract_ema_state_dict(state_dict)

        # Assert
        assert "network.linear.weight" in result
        assert "network.linear.bias" in result
        torch.testing.assert_close(
            result["network.linear.weight"],
            torch.tensor([1.0]),
        )
        torch.testing.assert_close(
            result["network.linear.bias"],
            torch.tensor([2.0]),
        )

    def test_prepare_filters_to_forecast_keys(
            self,
    ) -> None:
        r'''Only forecast-module keys survive preparation.'''
        # Arrange
        instance = _build_simple_forecast()
        checkpoint = _make_training_checkpoint(
            instance,
            network_weight=3.0,
            network_bias=4.0,
            ema_weight=9.0,
            ema_bias=10.0,
        )

        # Act
        result = _prepare_checkpoint_state_dict(
            checkpoint=checkpoint,
            forecast_module=instance,
            load_ema=False,
        )

        # Assert -- only keys in the forecast module
        forecast_keys = set(
            instance.state_dict().keys()
        )
        assert set(result.keys()) <= forecast_keys
        # EMA keys must not appear
        for key in result:
            assert not key.startswith("ema_network.")


# -----------------------------------------------------------
# Error / warning tests
# -----------------------------------------------------------

class TestCheckpointErrors:
    r'''Error and warning condition tests.'''

    def test_no_checkpoint_skips_load(self) -> None:
        r'''ckpt_path=None means torch.load not called.'''
        # Arrange
        cfg = _model_cfg(ckpt_path=None)
        instance = _build_simple_forecast()

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch("torch.load") as mock_load:
                load_forecast_model(
                    cfg,
                    fabric=_make_test_fabric(),
                )

                # Assert
                mock_load.assert_not_called()

    def test_load_ema_raises_without_ema_weights(
            self,
    ) -> None:
        r'''ValueError raised when no EMA state present.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
            load_ema=True,
        )
        instance = _build_simple_forecast()
        checkpoint = _make_training_checkpoint(
            instance,
            network_weight=3.0,
            network_bias=4.0,
        )

        # Act & Assert
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=checkpoint,
            ):
                with pytest.raises(
                    ValueError,
                    match="EMA weights",
                ):
                    load_forecast_model(
                        cfg,
                        fabric=_make_test_fabric(),
                    )

    def test_missing_keys_warning(
            self,
            caplog: pytest.LogCaptureFixture,
    ) -> None:
        r'''Missing checkpoint keys logged at WARNING.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
        )
        instance = _build_simple_forecast()
        empty_state_dict: dict = {}

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=empty_state_dict,
            ):
                with caplog.at_level(logging.WARNING):
                    load_forecast_model(
                        cfg,
                        fabric=_make_test_fabric(),
                    )

        # Assert
        assert any(
            "missing" in r.message.lower()
            for r in caplog.records
        )

    def test_unexpected_checkpoint_keys_do_not_warn(
            self,
            caplog: pytest.LogCaptureFixture,
    ) -> None:
        r'''Unused checkpoint keys cause no warning.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
        )
        instance = _build_simple_forecast()
        extra_state_dict = dict(instance.state_dict())
        extra_state_dict[
            "nonexistent_layer.weight"
        ] = torch.zeros(1)

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=extra_state_dict,
            ):
                with caplog.at_level(logging.WARNING):
                    load_forecast_model(
                        cfg,
                        fabric=_make_test_fabric(),
                    )

        # Assert
        assert len(caplog.records) == 0

    def test_unexpected_keys_warning_logged(
            self,
            caplog: pytest.LogCaptureFixture,
    ) -> None:
        r'''Unexpected keys in load_state_dict trigger warning.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
        )
        instance = _build_simple_forecast()
        fake_state_dict = dict(instance.state_dict())

        mock_result = MagicMock()
        mock_result.missing_keys = []
        mock_result.unexpected_keys = [
            "extra.weight",
        ]

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=fake_state_dict,
            ):
                with patch.object(
                    type(instance),
                    "load_state_dict",
                    return_value=mock_result,
                ):
                    with caplog.at_level(
                        logging.WARNING
                    ):
                        load_forecast_model(
                            cfg,
                            fabric=_make_test_fabric(),
                        )

        # Assert
        assert any(
            "unexpected" in r.message.lower()
            for r in caplog.records
        )


# -----------------------------------------------------------
# Helper for compiled-key checkpoints
# -----------------------------------------------------------

def _make_compiled_training_checkpoint(
        instance: SimpleForecast,
        network_weight: float = 1.0,
        network_bias: float = 0.0,
        ema_weight: Optional[float] = None,
        ema_bias: Optional[float] = None,
) -> Dict[str, Any]:
    r'''Return a training checkpoint with _orig_mod keys.

    Simulates a checkpoint saved from a torch.compile-wrapped
    training run where state_dict keys contain the
    ``._orig_mod`` infix.

    Parameters
    ----------
    instance : SimpleForecast
        Module whose state_dict provides the key layout.
    network_weight : float, optional
        Fill value for network weight. Default is 1.0.
    network_bias : float, optional
        Fill value for network bias. Default is 0.0.
    ema_weight : float or None, optional
        Fill value for EMA weight. Default is None.
    ema_bias : float or None, optional
        Fill value for EMA bias. Default is None.

    Returns
    -------
    Dict[str, Any]
        Training-style checkpoint with ``_orig_mod`` keys.
    '''
    state_dict: Dict[str, Any] = {}
    for key, value in instance.state_dict().items():
        compiled_key = key.replace(
            "network.", "network._orig_mod.",
        )
        state_dict[compiled_key] = value.clone()
    state_dict[
        "network._orig_mod.linear.weight"
    ] = torch.full_like(
        instance.network.linear.weight,
        fill_value=network_weight,
    )
    state_dict[
        "network._orig_mod.linear.bias"
    ] = torch.full_like(
        instance.network.linear.bias,
        fill_value=network_bias,
    )
    if ema_weight is not None:
        state_dict[
            "ema_network.module._orig_mod"
            ".linear.weight"
        ] = torch.full_like(
            instance.network.linear.weight,
            fill_value=ema_weight,
        )
        state_dict[
            "ema_network.module._orig_mod"
            ".linear.bias"
        ] = torch.full_like(
            instance.network.linear.bias,
            fill_value=(
                0.0 if ema_bias is None else ema_bias
            ),
        )
        state_dict[
            "ema_network.n_averaged"
        ] = torch.tensor(8)
    return {"state_dict": state_dict}


# -----------------------------------------------------------
# Compiled checkpoint unit tests
# -----------------------------------------------------------

class TestCompiledCheckpointUnittest:
    r'''Unit tests for _strip_compiled_prefix helper.'''

    def test_strip_compiled_prefix_cleans_keys(
            self,
    ) -> None:
        r'''_strip_compiled_prefix removes _orig_mod from keys.'''
        from {{cookiecutter.project_slug}}.forecast.checkpoint import (
            _strip_compiled_prefix,
        )

        # Arrange
        state_dict = {
            "network._orig_mod.linear.weight": (
                torch.tensor([1.0])
            ),
            "network._orig_mod.linear.bias": (
                torch.tensor([2.0])
            ),
        }

        # Act
        result = _strip_compiled_prefix(state_dict)

        # Assert
        assert "network.linear.weight" in result
        assert "network.linear.bias" in result
        assert all(
            "._orig_mod" not in k for k in result
        )

    def test_strip_compiled_prefix_noop_on_clean_keys(
            self,
    ) -> None:
        r'''_strip_compiled_prefix is a no-op for clean keys.'''
        from {{cookiecutter.project_slug}}.forecast.checkpoint import (
            _strip_compiled_prefix,
        )

        # Arrange
        state_dict = {
            "network.linear.weight": (
                torch.tensor([1.0])
            ),
            "network.linear.bias": (
                torch.tensor([2.0])
            ),
        }

        # Act
        result = _strip_compiled_prefix(state_dict)

        # Assert
        assert set(result.keys()) == set(
            state_dict.keys()
        )
        torch.testing.assert_close(
            result["network.linear.weight"],
            state_dict["network.linear.weight"],
        )


# -----------------------------------------------------------
# Compiled checkpoint functional tests
# -----------------------------------------------------------

class TestCompiledCheckpointFunctional:
    r'''End-to-end tests for loading compiled checkpoints.'''

    def test_loads_compiled_checkpoint(self) -> None:
        r'''Compiled checkpoint keys load into forecast module.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
        )
        instance = _build_simple_forecast()
        checkpoint = _make_compiled_training_checkpoint(
            instance,
            network_weight=3.5,
            network_bias=4.5,
        )

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=checkpoint,
            ):
                model = load_forecast_model(
                    cfg,
                    fabric=_make_test_fabric(),
                )

        # Assert
        torch.testing.assert_close(
            model.module.network.linear.weight,
            torch.full_like(
                model.module.network.linear.weight,
                fill_value=3.5,
            ),
        )
        torch.testing.assert_close(
            model.module.network.linear.bias,
            torch.full_like(
                model.module.network.linear.bias,
                fill_value=4.5,
            ),
        )

    def test_loads_compiled_ema_checkpoint(
            self,
    ) -> None:
        r'''Compiled EMA checkpoint keys load correctly.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
            load_ema=True,
        )
        instance = _build_simple_forecast()
        checkpoint = _make_compiled_training_checkpoint(
            instance,
            network_weight=-1.0,
            network_bias=-2.0,
            ema_weight=5.5,
            ema_bias=6.5,
        )

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=checkpoint,
            ):
                model = load_forecast_model(
                    cfg,
                    fabric=_make_test_fabric(),
                )

        # Assert
        torch.testing.assert_close(
            model.module.network.linear.weight,
            torch.full_like(
                model.module.network.linear.weight,
                fill_value=5.5,
            ),
        )
        torch.testing.assert_close(
            model.module.network.linear.bias,
            torch.full_like(
                model.module.network.linear.bias,
                fill_value=6.5,
            ),
        )

    def test_compiled_checkpoint_no_missing_keys_warning(
            self,
            caplog: pytest.LogCaptureFixture,
    ) -> None:
        r'''No missing-keys warning for compiled checkpoints.'''
        # Arrange
        cfg = _model_cfg(
            ckpt_path="/fake/path.ckpt",
        )
        instance = _build_simple_forecast()
        checkpoint = _make_compiled_training_checkpoint(
            instance,
            network_weight=1.0,
            network_bias=0.0,
        )

        # Act
        with patch(
            f"{_CKPT}.instantiate",
            return_value=instance,
        ):
            with patch(
                "torch.load",
                return_value=checkpoint,
            ):
                with caplog.at_level(logging.WARNING):
                    load_forecast_model(
                        cfg,
                        fabric=_make_test_fabric(),
                    )

        # Assert -- no missing keys warning
        assert not any(
            "missing" in r.message.lower()
            for r in caplog.records
        )
