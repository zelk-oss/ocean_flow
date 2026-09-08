# -*- coding: utf-8 -*-
r'''Tests for ocean_flow/modules/forecast.py – FlowMatchingForecastModule.'''

# External modules
import numpy as np
import pytest
import torch

# Internal modules
from flow_core.pre_processing import PrePipeline
from flow_core.post_processing import PostPipeline
from flow_forecast.forecast.forecast_model import ForecastModel
from ocean_flow.modules.forecast import FlowMatchingForecastModule
from tests.conftest import (
    IdentityPostModule,
    IdentityPreModule,
    make_fabric,
)


# -----------------------------------------------------------
# Helper classes
# -----------------------------------------------------------

class _TimeConditionedVelocityNetwork(torch.nn.Module):
    r'''Velocity network accepting a positional time tensor.'''

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor | None = None,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''Return the first input channel as the velocity.'''
        return x[:, :1]


def _build_module(n_int: int = 1) -> FlowMatchingForecastModule:
    r'''Return a FlowMatchingForecastModule with identity pipelines.'''
    return FlowMatchingForecastModule(
        network=_TimeConditionedVelocityNetwork(),
        pre_pipeline=PrePipeline(states_surface=IdentityPreModule()),
        post_pipeline=PostPipeline(states_surface=IdentityPostModule()),
        n_int=n_int,
    )


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestFlowMatchingForecastModuleFunctional:
    r'''Functional tests for FlowMatchingForecastModule.'''

    def test_forward_accepts_named_state(self) -> None:
        r'''forward accepts a channel-less state keyword, returns (tensor,).'''
        module = _build_module()
        state = torch.zeros((2, 1, 4, 8))

        out = module.forward(q=state)

        assert isinstance(out, tuple)
        assert len(out) == 1
        assert out[0].shape == (2, 1, 4, 8)

    def test_forward_accepts_channel_state(self) -> None:
        r'''forward accepts a (B, n_in_steps, C, H, W) state.'''
        module = _build_module()
        state = torch.zeros((3, 2, 1, 4, 8))

        out = module.forward(q=state)

        assert out[0].shape == (3, 1, 1, 4, 8)

    def test_forward_uses_last_time_step(self) -> None:
        r'''forward conditions the flow on only the last time step.'''

        class _CapturePreModule(torch.nn.Module):
            r'''Pre module recording the tensor it receives.'''

            def __init__(self) -> None:
                super().__init__()
                self.last_input: torch.Tensor | None = None

            def forward(
                    self,
                    in_tensor: torch.Tensor,
                    *args: object,
                    **kwargs: object,
            ) -> torch.Tensor:
                r'''Record and pass through the input unchanged.'''
                self.last_input = in_tensor
                return in_tensor

        capture = _CapturePreModule()
        module = FlowMatchingForecastModule(
            network=_TimeConditionedVelocityNetwork(),
            pre_pipeline=PrePipeline(states_surface=capture),
            post_pipeline=PostPipeline(states_surface=IdentityPostModule()),
            n_int=1,
        )
        state = torch.zeros((2, 3, 4, 8))
        state[:, -1] = 5.0

        module.forward(q=state)

        torch.testing.assert_close(capture.last_input, state[:, -1:])

    def test_sample_residual_with_flow_returns_matching_shape(self) -> None:
        r'''sample_residual_with_flow returns a (B, C, H, W) residual.'''
        module = _build_module(n_int=2)
        condition = torch.zeros((3, 1, 4, 8))

        residual = module.sample_residual_with_flow(condition)

        assert residual.shape == (3, 1, 4, 8)

    def test_forward_accepts_positional_state(self) -> None:
        r'''forward accepts state positionally without kwargs.'''
        module = _build_module()
        state = torch.zeros((2, 1, 4, 8))

        out = module.forward(state)

        assert out[0].shape == (2, 1, 4, 8)

    def test_compatible_with_forecast_model(self) -> None:
        r"""forward's tuple/shape output satisfies ForecastModel."""
        module = _build_module()
        forecast_model = ForecastModel(
            module=module,
            fabric=make_fabric(),
            dtype=torch.float32,
            compile=False,
        )
        forecast_model.set_state({
            "q": np.zeros((2, 1, 4, 8), dtype=np.float32),
        })

        trajectory = forecast_model.advance(n=1)

        assert trajectory["q"].shape == (2, 1, 4, 8)


# -----------------------------------------------------------
# Error tests
# -----------------------------------------------------------

class TestFlowMatchingForecastModuleErrors:
    r'''Error condition tests for FlowMatchingForecastModule.'''

    def test_init_raises_on_nonpositive_n_int(self) -> None:
        r'''__init__ raises ValueError for n_int <= 0.'''
        with pytest.raises(ValueError, match="n_int"):
            FlowMatchingForecastModule(
                network=_TimeConditionedVelocityNetwork(),
                pre_pipeline=PrePipeline(
                    states_surface=IdentityPreModule(),
                ),
                post_pipeline=PostPipeline(
                    states_surface=IdentityPostModule(),
                ),
                n_int=0,
            )

    def test_sample_residual_raises_on_invalid_ndim(self) -> None:
        r'''sample_residual_with_flow rejects a non-4D tensor.'''
        module = _build_module()
        condition = torch.zeros((4, 8))

        with pytest.raises(ValueError, match="Expected condition"):
            module.sample_residual_with_flow(condition)

    def test_forward_raises_without_state(self) -> None:
        r'''forward raises when no state kwarg is given.'''
        module = _build_module()

        with pytest.raises(ValueError, match="exactly one"):
            module.forward()

    def test_forward_raises_on_state_and_kwargs(self) -> None:
        r'''forward raises when state and kwargs are both given.'''
        module = _build_module()
        state = torch.zeros((2, 1, 4, 8))

        with pytest.raises(ValueError, match="both a positional"):
            module.forward(state, q=state)

    def test_forward_raises_on_invalid_ndim(self) -> None:
        r'''forward rejects a non-4D/5D state tensor.'''
        module = _build_module()
        state = torch.zeros((4, 8))

        with pytest.raises(ValueError, match="Expected state"):
            module.forward(q=state)
