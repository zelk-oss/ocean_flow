# -*- coding: utf-8 -*-
r'''Tests for ocean_flow/networks/unet.py.

Covers the vendored Unet architecture and its building blocks.
'''

# External modules
import torch

# Internal modules
from ocean_flow.networks import unet


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestUnetFunctional:
    r'''End-to-end tests for the Unet network.'''

    def test_forward_without_classes(self) -> None:
        r'''Unet forward runs and returns the expected shape.'''
        net = unet.Unet(
            num_classes=1,
            in_channels=2,
            out_channels=1,
            dim=4,
            dim_mults=(1, 2),
            resnet_block_groups=2,
            learned_sinusoidal_cond=True,
            attn_dim_head=4,
            attn_heads=1,
            use_classes=False,
        )
        x = torch.zeros((2, 2, 8, 8))
        t = torch.zeros((2,))

        out = net(x, t)

        assert out.shape == (2, 1, 8, 8)

    def test_forward_with_classes(self) -> None:
        r'''Unet forward accepts a classes tensor when use_classes=True.'''
        net = unet.Unet(
            num_classes=3,
            in_channels=2,
            out_channels=1,
            dim=4,
            dim_mults=(1, 2),
            resnet_block_groups=2,
            learned_sinusoidal_cond=True,
            attn_dim_head=4,
            attn_heads=1,
            use_classes=True,
        )
        x = torch.zeros((2, 2, 8, 8))
        t = torch.zeros((2,))
        classes = torch.zeros((2,), dtype=torch.long)

        out = net(x, t, classes=classes)

        assert out.shape == (2, 1, 8, 8)

    def test_forward_with_random_fourier_features(self) -> None:
        r'''Unet forward runs with random_fourier_features=True.'''
        net = unet.Unet(
            num_classes=1,
            in_channels=2,
            out_channels=1,
            dim=4,
            dim_mults=(1, 2),
            resnet_block_groups=2,
            learned_sinusoidal_cond=False,
            random_fourier_features=True,
            attn_dim_head=4,
            attn_heads=1,
            use_classes=False,
        )
        x = torch.zeros((2, 2, 8, 8))
        t = torch.zeros((2,))

        out = net(x, t)

        assert out.shape == (2, 1, 8, 8)

    def test_forward_with_plain_sinusoidal_embedding(self) -> None:
        r'''Unet forward runs with the non-learned sinusoidal embedding.'''
        net = unet.Unet(
            num_classes=1,
            in_channels=2,
            out_channels=1,
            dim=4,
            dim_mults=(1, 2),
            resnet_block_groups=2,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            attn_dim_head=4,
            attn_heads=1,
            use_classes=False,
        )
        x = torch.zeros((2, 2, 8, 8))
        t = torch.zeros((2,))

        out = net(x, t)

        assert out.shape == (2, 1, 8, 8)


# -----------------------------------------------------------
# Unit tests
# -----------------------------------------------------------

class TestUnetHelpersUnittest:
    r'''Isolated unit tests for small helper functions/classes.'''

    def test_exists_true_and_false(self) -> None:
        r'''exists returns True/False for present/absent values.'''
        assert unet.exists(0) is True
        assert unet.exists(None) is False

    def test_default_returns_value_when_present(self) -> None:
        r'''default returns val when it exists.'''
        assert unet.default(5, 10) == 5

    def test_default_calls_callable_fallback(self) -> None:
        r'''default calls d() when val is None and d is callable.'''
        assert unet.default(None, lambda: 7) == 7

    def test_default_returns_plain_fallback(self) -> None:
        r'''default returns d directly when val is None and d is not callable.'''
        assert unet.default(None, 9) == 9

    def test_identity_returns_input_unchanged(self) -> None:
        r'''identity returns its first argument unchanged.'''
        tensor = torch.ones((2,))
        assert unet.identity(tensor, 1, key="value") is tensor

    def test_upsample_without_dim_out(self) -> None:
        r'''Upsample defaults dim_out to dim when omitted.'''
        layer = unet.Upsample(4)
        x = torch.zeros((1, 4, 4, 4))

        out = layer(x)

        assert out.shape == (1, 4, 8, 8)

    def test_downsample_without_dim_out(self) -> None:
        r'''Downsample defaults dim_out to dim when omitted.'''
        layer = unet.Downsample(4)
        x = torch.zeros((1, 4, 8, 8))

        out = layer(x)

        assert out.shape == (1, 4, 4, 4)

    def test_residual_adds_input(self) -> None:
        r'''Residual adds its wrapped function's output to the input.'''
        layer = unet.Residual(lambda x: x * 0)
        x = torch.full((1, 2, 2, 2), 3.0)

        out = layer(x)

        torch.testing.assert_close(out, x)

    def test_rmsnorm_forward_shape(self) -> None:
        r'''RMSNorm forward preserves input shape.'''
        layer = unet.RMSNorm(4)
        x = torch.randn((1, 4, 3, 3))

        out = layer(x)

        assert out.shape == x.shape

    def test_prenorm_applies_norm_then_fn(self) -> None:
        r'''PreNorm normalizes before applying the wrapped function.'''
        layer = unet.PreNorm(4, lambda x: x)
        x = torch.randn((1, 4, 3, 3))

        out = layer(x)

        assert out.shape == x.shape

    def test_sinusoidal_pos_emb_shape(self) -> None:
        r'''SinusoidalPosEmb returns an embedding of the expected size.'''
        layer = unet.SinusoidalPosEmb(8)
        t = torch.zeros((3,))

        out = layer(t)

        assert out.shape == (3, 8)

    def test_random_or_learned_sinusoidal_pos_emb_shape(self) -> None:
        r'''RandomOrLearnedSinusoidalPosEmb returns the expected shape.'''
        layer = unet.RandomOrLearnedSinusoidalPosEmb(8, is_random=True)
        t = torch.zeros((3,))

        out = layer(t)

        assert out.shape == (3, 9)

    def test_block_forward_without_scale_shift(self) -> None:
        r'''Block forward runs without a scale_shift tuple.'''
        layer = unet.Block(2, 4, groups=2)
        x = torch.randn((1, 2, 4, 4))

        out = layer(x)

        assert out.shape == (1, 4, 4, 4)

    def test_block_forward_with_scale_shift(self) -> None:
        r'''Block forward applies an explicit scale_shift tuple.'''
        layer = unet.Block(2, 4, groups=2)
        x = torch.randn((1, 2, 4, 4))
        scale = torch.zeros((1, 4, 1, 1))
        shift = torch.zeros((1, 4, 1, 1))

        out = layer(x, scale_shift=(scale, shift))

        assert out.shape == (1, 4, 4, 4)

    def test_resnet_block_without_conditioning(self) -> None:
        r'''ResnetBlock forward skips the mlp path with no conditioning.'''
        block = unet.ResnetBlock(2, 4, time_emb_dim=6, groups=2)
        x = torch.randn((1, 2, 4, 4))

        out = block(x)

        assert out.shape == (1, 4, 4, 4)

    def test_resnet_block_with_time_and_class_conditioning(self) -> None:
        r'''ResnetBlock forward applies both time and class conditioning.'''
        block = unet.ResnetBlock(
            2, 4, time_emb_dim=6, classes_emb_dim=3, groups=2,
        )
        x = torch.randn((1, 2, 4, 4))
        time_emb = torch.randn((1, 6))
        class_emb = torch.randn((1, 3))

        out = block(x, time_emb=time_emb, class_emb=class_emb)

        assert out.shape == (1, 4, 4, 4)

    def test_resnet_block_matching_dims_uses_identity_shortcut(
            self,
    ) -> None:
        r'''ResnetBlock uses an identity shortcut when dims match.'''
        block = unet.ResnetBlock(4, 4, time_emb_dim=6, groups=2)
        assert isinstance(block.res_conv, torch.nn.Identity)

    def test_linear_attention_forward_shape(self) -> None:
        r'''LinearAttention forward preserves spatial shape.'''
        layer = unet.LinearAttention(4, heads=2, dim_head=4)
        x = torch.randn((1, 4, 4, 4))

        out = layer(x)

        assert out.shape == x.shape

    def test_attention_forward_shape(self) -> None:
        r'''Attention forward preserves spatial shape.'''
        layer = unet.Attention(4, heads=2, dim_head=4)
        x = torch.randn((1, 4, 4, 4))

        out = layer(x)

        assert out.shape == x.shape
