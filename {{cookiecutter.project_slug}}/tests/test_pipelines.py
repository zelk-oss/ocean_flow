# -*- coding: utf-8 -*-
r'''Issue #14 tests for src/pipelines package.'''

# External modules
import torch

# Internal modules
from {{cookiecutter.project_slug}}.pipelines import (
    LowerBoundPrediction,
    PostPipeline,
    PreNormalization,
    PrePipeline,
    TendencyPrediction,
)
from {{cookiecutter.project_slug}}.pipelines.utils import add_dimensions


class _PreShift(torch.nn.Module):
    r'''Pre module that shifts tensor by a constant value.'''

    def __init__(self, shift: float) -> None:
        super().__init__()
        self.shift = shift

    def forward(
        self,
        in_tensor: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return in_tensor + self.shift


class _PostScale(torch.nn.Module):
    r'''Post module with reversible linear transform.'''

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def to_latent(
        self,
        target: torch.Tensor,
        initial: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return target / self.scale

    def forward(
        self,
        prediction: torch.Tensor,
        initial: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return prediction * self.scale


class TestPipelineFunctional:
    r'''End-to-end pipeline tests.'''

    def test_pre_pipeline_applies_modules_in_insertion_order(self):
        r'''PrePipeline applies layers in provided insertion order.'''
        pipeline = PrePipeline(
            first=_PreShift(1.0),
            second=_PreShift(2.0),
            third=_PreShift(3.0),
        )
        in_tensor = torch.zeros(
            (1, 2, 2), dtype=torch.float32,
        )

        out = pipeline(in_tensor)

        torch.testing.assert_close(
            out, torch.full_like(in_tensor, 6.0),
        )

    def test_post_pipeline_forward_applies_modules_in_insertion_order(self):
        r'''PostPipeline.forward applies layers in insertion order.'''
        pipeline = PostPipeline(
            first=_PostScale(2.0),
            second=_PostScale(5.0),
        )
        prediction = torch.full((1, 2, 2), 3.0)

        out = pipeline(
            prediction,
            initial=torch.zeros_like(prediction),
        )

        torch.testing.assert_close(
            out, torch.full_like(prediction, 30.0),
        )

    def test_post_pipeline_to_latent_applies_reverse_order(self):
        r'''PostPipeline.to_latent applies inverse chain in reverse order.'''
        pipeline = PostPipeline(
            first=_PostScale(2.0),
            second=_PostScale(5.0),
        )
        target = torch.full((1, 2, 2), 30.0)

        latent = pipeline.to_latent(
            target, initial=torch.zeros_like(target),
        )

        torch.testing.assert_close(
            latent, torch.full_like(target, 3.0),
        )

    def test_tendency_prediction_to_latent_and_forward_are_inverse(self):
        r'''`to_latent` and `forward` compose to identity on targets.'''
        module = TendencyPrediction(
            std=[2.0, 4.0], add_dims=(1, 2),
        )
        initial = torch.tensor(
            [[[1.0]], [[3.0]]], dtype=torch.float32,
        ).unsqueeze(0)
        target = torch.tensor(
            [[[5.0]], [[11.0]]], dtype=torch.float32,
        ).unsqueeze(0)

        latent = module.to_latent(
            target=target, initial=initial,
        )
        recovered = module(
            prediction=latent, initial=initial,
        )

        torch.testing.assert_close(recovered, target)

    def test_pre_normalization_registers_buffers_and_normalizes(self):
        r'''PreNormalization stores buffers and computes normalized tensor.'''
        module = PreNormalization(
            mean=[1.0, 2.0],
            std=[2.0, 4.0],
            add_dims=(1, 2),
            epsilon=1e-8,
        )
        in_tensor = torch.tensor(
            [[[[3.0]], [[10.0]]]],
            dtype=torch.float32,
        )

        out = module(in_tensor)

        assert "mean" in dict(module.named_buffers())
        assert "std" in dict(module.named_buffers())
        assert module.mean.shape == (2, 1, 1)
        assert module.std.shape == (2, 1, 1)
        expected = torch.tensor(
            [[[[1.0]], [[2.0]]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            out, expected, rtol=1e-6, atol=1e-6,
        )


class TestPipelineUnittest:
    r'''Isolated unit tests for pipeline modules and utilities.'''

    def test_add_dimensions_inserts_singletons_in_sorted_order(self):
        r'''Dimension insertion order is sorted internally.'''
        tensor = torch.ones(
            (2, 3), dtype=torch.float32,
        )

        out = add_dimensions(tensor, dims=(2, 0))

        assert out.shape == (1, 2, 1, 3)

    def test_add_dimensions_supports_non_default_dims(self):
        r'''Non-default `dims` values create expected broadcast shape.'''
        tensor = torch.ones(
            (2,), dtype=torch.float32,
        )

        out = add_dimensions(
            tensor, dims=(0, 2, 3),
        )

        assert out.shape == (1, 2, 1, 1)

    def test_lower_bound_forward_clamps_with_broadcasted_bounds(self):
        r'''Lower bound clamp handles mixed-sign predictions correctly.'''
        module = LowerBoundPrediction(
            lower_bound=[-1.0, 0.5],
            add_dims=(1, 2),
        )
        prediction = torch.tensor(
            [
                [
                    [[-2.0, -0.5]],
                    [[0.0, 2.0]],
                ]
            ],
            dtype=torch.float32,
        )

        out = module(
            prediction=prediction,
            initial=torch.zeros_like(prediction),
        )

        expected = torch.tensor(
            [
                [
                    [[-1.0, -0.5]],
                    [[0.5, 2.0]],
                ]
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(out, expected)

    def test_lower_bound_to_latent_is_passthrough(self):
        r'''LowerBoundPrediction.to_latent returns target unchanged.'''
        module = LowerBoundPrediction(
            lower_bound=[0.0, 0.0],
            add_dims=(1, 2),
        )
        target = torch.randn(2, 2, 3, 4)

        latent = module.to_latent(
            target=target,
            initial=torch.zeros_like(target),
        )

        torch.testing.assert_close(latent, target)

    def test_pre_normalization_non_default_add_dims(self):
        r'''PreNormalization supports non-default `add_dims` values.'''
        module = PreNormalization(
            mean=[1.0, 2.0],
            std=[1.0, 2.0],
            add_dims=(0, 2),
        )
        in_tensor = torch.tensor(
            [[[[2.0], [4.0]]]],
            dtype=torch.float32,
        )

        out = module(in_tensor)

        assert module.mean.shape == (1, 2, 1)
        assert out.shape == in_tensor.shape
