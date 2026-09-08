# -*- coding: utf-8 -*-
r'''Tests for ocean_flow/modules/training.py – FlowMatchingTrainingModule.'''

# External modules
import pytest
import torch

# Internal modules
from flow_core.pre_processing import PrePipeline
from flow_core.post_processing import PostPipeline
from ocean_flow.modules.training import FlowMatchingTrainingModule


# -----------------------------------------------------------
# Helper classes
# -----------------------------------------------------------

class _LatentRecorder(torch.nn.Module):
    r'''Post-processing helper recording to_latent calls.'''

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def to_latent(
            self,
            target: torch.Tensor,
            initial: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''Record target/initial and return difference.'''
        self.calls.append((target.clone(), initial.clone()))
        return target - initial

    def forward(
            self,
            prediction: torch.Tensor,
            initial: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''Pass through prediction unchanged.'''
        return prediction


class _ScalingPreModule(torch.nn.Module):
    r'''Pre module scaling the input by a fixed factor.'''

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(
            self,
            in_tensor: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''Scale input tensor by factor.'''
        return in_tensor * self.factor


class _CaptureNetwork(torch.nn.Module):
    r'''Network recording the last input it received.'''

    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(
            self,
            x: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''Record input and return a zero velocity prediction.'''
        self.last_input = x
        return torch.zeros(
            x.shape[0], 1, x.shape[-2], x.shape[-1],
            device=x.device, dtype=x.dtype,
        )


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestFlowMatchingTrainingModuleFunctional:
    r'''Functional tests for FlowMatchingTrainingModule.estimate_loss.'''

    def test_estimate_loss_uses_pipelines(
            self,
            monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        r'''estimate_loss applies pre_pipeline and post_pipeline logic.'''
        monkeypatch.setattr(
            FlowMatchingTrainingModule,
            "_assert_normalized",
            staticmethod(lambda *args, **kwargs: None),
        )

        recorder = _LatentRecorder()
        module = FlowMatchingTrainingModule(
            network=_CaptureNetwork(),
            pre_pipeline=PrePipeline(
                states_surface=_ScalingPreModule(2.0),
            ),
            post_pipeline=PostPipeline(
                states_surface=recorder,
            ),
            ema_rate=0.0,
        )

        state = torch.ones((2, 1, 4, 8), dtype=torch.float32)
        residual = torch.full((2, 1, 4, 8), 2.0, dtype=torch.float32)
        batch = {"input": state, "residual": residual}

        out = module.estimate_loss(batch, prefix="train")

        assert "loss" in out
        assert module.network.last_input is not None
        torch.testing.assert_close(
            module.network.last_input[:, 1:],
            state * 2.0,
        )

        expected_target = state + residual
        recorded_target, recorded_initial = recorder.calls[0]
        torch.testing.assert_close(recorded_target, expected_target)
        torch.testing.assert_close(recorded_initial, state)

    def test_training_step_delegates_to_estimate_loss(
            self,
            monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        r'''training_step calls estimate_loss and returns its output.'''
        monkeypatch.setattr(
            FlowMatchingTrainingModule,
            "_assert_normalized",
            staticmethod(lambda *args, **kwargs: None),
        )
        module = FlowMatchingTrainingModule(
            network=_CaptureNetwork(),
            pre_pipeline=PrePipeline(
                states_surface=_ScalingPreModule(1.0),
            ),
            post_pipeline=PostPipeline(
                states_surface=_LatentRecorder(),
            ),
            ema_rate=0.0,
        )
        state = torch.ones((2, 1, 4, 8), dtype=torch.float32)
        residual = torch.zeros((2, 1, 4, 8), dtype=torch.float32)

        out = module.training_step(
            batch={"input": state, "residual": residual}, batch_idx=0,
        )

        assert "loss" in out


# -----------------------------------------------------------
# Unit tests
# -----------------------------------------------------------

class TestAssertNormalizedUnittest:
    r'''Isolated unit tests for FlowMatchingTrainingModule._assert_normalized.'''

    def test_accepts_normalized_tensor(self) -> None:
        r'''_assert_normalized passes for mean~0/std~1 tensors.'''
        rng = torch.Generator().manual_seed(19921225)
        raw = torch.randn((1000,), generator=rng)
        tensor = (raw - raw.mean()) / raw.std()

        FlowMatchingTrainingModule._assert_normalized(tensor, "latent")


# -----------------------------------------------------------
# Error tests
# -----------------------------------------------------------

class TestAssertNormalizedErrors:
    r'''Error condition tests for FlowMatchingTrainingModule._assert_normalized.'''

    def test_raises_on_non_finite(self) -> None:
        r'''_assert_normalized raises for non-finite values.'''
        tensor = torch.tensor([0.0, float("nan"), 0.0])

        with pytest.raises(AssertionError, match="non-finite"):
            FlowMatchingTrainingModule._assert_normalized(tensor, "latent")

    def test_raises_on_bad_mean(self) -> None:
        r'''_assert_normalized raises when mean is far from 0.'''
        tensor = torch.full((100,), 5.0)

        with pytest.raises(AssertionError, match="mean is not close"):
            FlowMatchingTrainingModule._assert_normalized(tensor, "latent")

    def test_raises_on_bad_std(self) -> None:
        r'''_assert_normalized raises when std is far from 1.'''
        tensor = torch.zeros((100,))

        with pytest.raises(AssertionError, match="std is not close"):
            FlowMatchingTrainingModule._assert_normalized(tensor, "latent")
