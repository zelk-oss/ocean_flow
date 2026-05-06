# -*- coding: utf-8 -*-
r'''Tests for src/modules package – ForecastModule,
TrainingModule, and utility functions.'''

# External modules
import pytest
import torch

# Internal modules
from ocean_flow.modules.forecast_module import ForecastModule
from ocean_flow.modules.train_module import TrainingModule
from ocean_flow.modules.utils import (
    preprocess_data,
    process_inputs,
    split_wd_params,
)
from ocean_flow.pipelines import PostPipeline, PrePipeline
from tests.conftest import (
    DummyNetwork,
    IdentityPostModule,
    IdentityPreModule,
)


# -----------------------------------------------------------
# Helper classes
# -----------------------------------------------------------

class _ForecastIncrementModule(ForecastModule):
    r'''Concrete forecast module that increments states.'''

    def __init__(
            self,
            network: torch.nn.Module,
            pre_pipeline: PrePipeline,
            post_pipeline: PostPipeline,
            surface_delta: float = 1.0,
    ) -> None:
        super().__init__(
            network=network,
            pre_pipeline=pre_pipeline,
            post_pipeline=post_pipeline,
        )
        self.surface_delta = surface_delta

    def forward(
            self,
            states_surface: torch.Tensor,
            states_levels: torch.Tensor,
            **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r'''Increment last-step states deterministically.'''
        surface = (
            states_surface[:, -1] + self.surface_delta
        )
        levels = states_levels[:, -1] + 10.0
        return surface, levels


class _LatentRecorder(torch.nn.Module):
    r'''Post-processing helper recording to_latent calls.'''

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []

    def to_latent(
            self,
            target: torch.Tensor,
            initial: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''Record target/initial and return difference.'''
        self.calls.append(
            (target.clone(), initial.clone()),
        )
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


class _ParamPreModule(torch.nn.Module):
    r'''Simple pre module with one learnable parameter.'''

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(1.0),
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''Scale input tensor by learned parameter.'''
        return in_tensor * self.scale


class _ParamPostModule(torch.nn.Module):
    r'''Simple post module with one learnable parameter.'''

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(
            torch.tensor(0.0),
        )

    def to_latent(
            self,
            target: torch.Tensor,
            initial: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''Return target unchanged.'''
        return target

    def forward(
            self,
            prediction: torch.Tensor,
            initial: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''Add bias to prediction.'''
        return prediction + self.bias


class _ProbeNet(torch.nn.Module):
    r'''Network exposing representative parameter types.'''

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.GroupNorm(1, 4)
        self.embedding = torch.nn.Embedding(4, 4)
        self.frozen = torch.nn.Parameter(
            torch.ones(4),
            requires_grad=False,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        r'''Forward through linear layer.'''
        return self.linear(x)


class _ProbeTrainingModule(TrainingModule):
    r'''TrainingModule for behavior testing.'''

    def __init__(
            self,
            *args: object,
            **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_prefixes: list[str] = []
        self.aux_prefixes: list[str] = []

    def estimate_loss(
            self,
            batch: dict,
            prefix: str = "train",
    ) -> dict:
        r'''Record prefix and return dummy loss.'''
        self.loss_prefixes.append(prefix)
        return {"loss": torch.tensor(1.0)}

    def estimate_auxiliary_losses(
            self,
            batch: dict,
            outputs: dict,
            prefix: str = "train",
    ) -> None:
        r'''Record auxiliary loss prefix.'''
        self.aux_prefixes.append(prefix)


class _BaseAuxTrainingModule(TrainingModule):
    r'''TrainingModule keeping base auxiliary-loss impl.'''

    def estimate_loss(
            self,
            batch: dict,
            prefix: str = "train",
    ) -> dict:
        r'''Return dummy loss.'''
        return {"loss": torch.tensor(1.0)}


class _MissingKeysModel(torch.nn.Module):
    r'''Model returning inconsistent parameter listings.'''

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.extra = torch.nn.Linear(2, 2)
        self._calls = 0

    def named_parameters(
            self,
            *args: object,
            **kwargs: object,
    ) -> object:
        r'''Return different params on each call.'''
        self._calls += 1
        if self._calls == 1:
            return iter([
                (
                    "linear.weight",
                    self.linear.weight,
                ),
            ])
        return iter([
            ("linear.weight", self.linear.weight),
            ("extra.weight", self.extra.weight),
        ])


class _IntersectingSetsModel(torch.nn.Module):
    r'''Model forcing same parameter into all sets.'''

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.group_norm = torch.nn.GroupNorm(1, 2)
        self.frozen = torch.nn.Parameter(
            torch.ones_like(self.linear.weight),
            requires_grad=False,
        )
        self._submodule_calls = 0

    def named_parameters(
            self,
            *args: object,
            **kwargs: object,
    ) -> object:
        r'''Return duplicated parameter names.'''
        return iter([
            ("linear.weight", self.linear.weight),
            ("linear.weight", self.linear.weight),
            ("linear.weight", self.frozen),
        ])

    def get_submodule(
            self,
            target: str,
    ) -> torch.nn.Module:
        r'''Return different modules on repeat calls.'''
        if target == "linear":
            self._submodule_calls += 1
            if self._submodule_calls == 2:
                return self.group_norm
            return self.linear
        return super().get_submodule(target)


# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def _build_forecast_module(
        surface_delta: float = 1.0,
) -> ForecastModule:
    r'''Return a deterministic ForecastModule for tests.'''
    return _ForecastIncrementModule(
        network=DummyNetwork(),
        pre_pipeline=PrePipeline(
            states_surface=IdentityPreModule(),
            states_levels=IdentityPreModule(),
        ),
        post_pipeline=PostPipeline(
            states_surface=IdentityPostModule(),
            states_levels=IdentityPostModule(),
        ),
        surface_delta=surface_delta,
    )


def _build_training_module(
        ema_rate: float = 0.0,
        lr: float = 1e-3,
        lr_warmup_steps: int = 5000,
        total_steps: int = 20,
) -> _ProbeTrainingModule:
    r'''Create probe training module with small network.'''
    network = _ProbeNet()
    pre_pipeline = PrePipeline(
        states_surface=_ParamPreModule(),
    )
    post_pipeline = PostPipeline(
        states_surface=_ParamPostModule(),
    )
    return _ProbeTrainingModule(
        network=network,
        pre_pipeline=pre_pipeline,
        post_pipeline=post_pipeline,
        ema_rate=ema_rate,
        weight_decay=0.1,
        total_steps=total_steps,
        lr=lr,
        lr_warmup_steps=lr_warmup_steps,
    )


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestForecastModuleFunctional:
    r'''Functional tests for ForecastModule abstraction.'''

    def test_stores_network(self) -> None:
        r'''__init__ stores network attribute.'''
        network = DummyNetwork()
        module = _ForecastIncrementModule(
            network=network,
            pre_pipeline=PrePipeline(
                states_surface=IdentityPreModule(),
            ),
            post_pipeline=PostPipeline(
                states_surface=IdentityPostModule(),
            ),
        )
        assert module.network is network

    def test_stores_pre_pipeline(self) -> None:
        r'''__init__ stores pre_pipeline attribute.'''
        pre = PrePipeline(
            states_surface=IdentityPreModule(),
        )
        module = _ForecastIncrementModule(
            network=DummyNetwork(),
            pre_pipeline=pre,
            post_pipeline=PostPipeline(
                states_surface=IdentityPostModule(),
            ),
        )
        assert module.pre_pipeline is pre

    def test_stores_post_pipeline(self) -> None:
        r'''__init__ stores post_pipeline attribute.'''
        post = PostPipeline(
            states_surface=IdentityPostModule(),
        )
        module = _ForecastIncrementModule(
            network=DummyNetwork(),
            pre_pipeline=PrePipeline(
                states_surface=IdentityPreModule(),
            ),
            post_pipeline=post,
        )
        assert module.post_pipeline is post

    def test_concrete_subclass_forward(self) -> None:
        r'''Concrete subclass forward returns incremented states.'''
        module = _build_forecast_module(surface_delta=2.0)
        surf = torch.zeros((1, 2, 2, 4, 8))
        surf[:, -1] = 3.0
        levels = torch.zeros((1, 2, 2, 3, 4, 8))
        levels[:, -1] = 5.0

        out_surf, out_levels = module.forward(
            states_surface=surf,
            states_levels=levels,
        )

        assert out_surf[0, 0, 0, 0].item() == pytest.approx(5.0)
        assert out_levels[0, 0, 0, 0, 0].item() == pytest.approx(15.0)


class TestModulesFunctional:
    r'''End-to-end tests for modules package.'''

    def test_training_step_delegates(self) -> None:
        r'''training_step calls estimate_loss with train.'''
        module = _build_training_module()
        out = module.training_step(
            batch={}, batch_idx=0,
        )
        assert module.loss_prefixes == ["train"]
        assert "loss" in out

    def test_validation_step_calls_both(
            self,
    ) -> None:
        r'''validation_step calls core and auxiliary loss.'''
        module = _build_training_module()
        out = module.validation_step(
            batch={}, batch_idx=0,
        )
        assert module.loss_prefixes == ["val"]
        assert module.aux_prefixes == ["val"]
        assert "loss" in out

    def test_configure_optimizers_uses_linear_warmup_then_cosine_decay(
            self,
    ) -> None:
        r'''configure_optimizers uses linear warmup then cosine decay.'''
        module = _build_training_module(
            lr=1.0,
            lr_warmup_steps=3,
            total_steps=6,
        )
        opt_cfg = module.configure_optimizers()
        optimizer = opt_cfg["optimizer"]
        scheduler = opt_cfg["lr_scheduler"][
            "scheduler"
        ]

        assert isinstance(
            optimizer, torch.optim.AdamW,
        )
        assert len(optimizer.param_groups) == 2
        assert not isinstance(
            scheduler,
            torch.optim.lr_scheduler
            .CosineAnnealingWarmRestarts,
        )

        recorded_lrs = [
            optimizer.param_groups[0]["lr"],
        ]
        for _ in range(module.total_steps):
            optimizer.step()
            scheduler.step()
            recorded_lrs.append(
                optimizer.param_groups[0]["lr"],
            )

        assert recorded_lrs[0] == pytest.approx(1e-3)
        assert recorded_lrs[1] > recorded_lrs[0]
        assert recorded_lrs[2] > recorded_lrs[1]
        assert recorded_lrs[module.lr_warmup_steps] == pytest.approx(1.0)
        assert recorded_lrs[module.lr_warmup_steps + 1] < recorded_lrs[
            module.lr_warmup_steps
        ]
        assert recorded_lrs[-1] == pytest.approx(1e-6)

        wd_group = optimizer.param_groups[0]
        nowd_group = optimizer.param_groups[1]
        assert wd_group["weight_decay"] == (
            pytest.approx(module.weight_decay)
        )
        assert nowd_group["weight_decay"] == (
            pytest.approx(0.0)
        )

        wd_ids = {
            id(p) for p in wd_group["params"]
        }
        nowd_ids = {
            id(p) for p in nowd_group["params"]
        }

        assert (
            id(module.network.linear.weight)
            in wd_ids
        )
        assert (
            id(module.network.linear.bias)
            in nowd_ids
        )
        assert (
            id(module.network.norm.weight)
            in nowd_ids
        )
        assert (
            id(module.network.embedding.weight)
            in nowd_ids
        )
        assert (
            id(
                module.pre_pipeline[
                    "states_surface"
                ].scale,
            )
            in nowd_ids
        )
        assert (
            id(
                module.post_pipeline[
                    "states_surface"
                ].bias,
            )
            in nowd_ids
        )

        frozen_id = id(module.network.frozen)
        assert frozen_id not in wd_ids
        assert frozen_id not in nowd_ids

    def test_configure_optimizers_uses_linear_warmup_only(
            self,
    ) -> None:
        r'''configure_optimizers uses only linear warmup if no remainder.'''
        module = _build_training_module(
            lr=1.0,
            lr_warmup_steps=4,
            total_steps=4,
        )
        opt_cfg = module.configure_optimizers()
        optimizer = opt_cfg["optimizer"]
        scheduler = opt_cfg["lr_scheduler"][
            "scheduler"
        ]

        assert isinstance(
            scheduler,
            torch.optim.lr_scheduler.LinearLR,
        )

        recorded_lrs = [
            optimizer.param_groups[0]["lr"],
        ]
        for _ in range(module.lr_warmup_steps):
            optimizer.step()
            scheduler.step()
            recorded_lrs.append(
                optimizer.param_groups[0]["lr"],
            )

        assert recorded_lrs[0] == pytest.approx(1e-3)
        assert recorded_lrs[-1] == pytest.approx(1.0)
        assert recorded_lrs[1] > recorded_lrs[0]
        assert recorded_lrs[2] > recorded_lrs[1]

    def test_configure_optimizers_uses_cosine_without_warmup(
            self,
    ) -> None:
        r'''configure_optimizers uses cosine schedule when warmup is zero.'''
        module = _build_training_module(
            lr=1.0,
            lr_warmup_steps=0,
            total_steps=5,
        )
        opt_cfg = module.configure_optimizers()
        optimizer = opt_cfg["optimizer"]
        scheduler = opt_cfg["lr_scheduler"][
            "scheduler"
        ]

        assert isinstance(
            scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )

        recorded_lrs = [
            optimizer.param_groups[0]["lr"],
        ]
        for _ in range(module.total_steps):
            optimizer.step()
            scheduler.step()
            recorded_lrs.append(
                optimizer.param_groups[0]["lr"],
            )

        assert recorded_lrs[0] == pytest.approx(1.0)
        assert recorded_lrs[1] < recorded_lrs[0]
        assert recorded_lrs[-1] == pytest.approx(1e-6)

    def test_process_inputs_contract(self) -> None:
        r'''Processed channels equal flattened time-var.'''
        states_surface = torch.ones(
            (2, 3, 2, 4, 8), dtype=torch.float32,
        )
        states_levels = torch.ones(
            (2, 3, 2, 3, 4, 8), dtype=torch.float32,
        )
        pre = PrePipeline(
            states_surface=IdentityPreModule(),
            states_levels=IdentityPreModule(),
        )

        out = process_inputs(
            states_surface=states_surface,
            states_levels=states_levels,
            pre_pipeline=pre,
        )

        assert out.shape == (2, 24, 4, 8)

    def test_preprocess_data_calls_to_latent(
            self,
    ) -> None:
        r'''Targets and initials to to_latent are correct.'''
        states_surface = torch.arange(
            2 * 4 * 2 * 2 * 2,
            dtype=torch.float32,
        ).reshape(2, 4, 2, 2, 2)
        states_levels = torch.arange(
            2 * 4 * 2 * 3 * 2 * 2,
            dtype=torch.float32,
        ).reshape(2, 4, 2, 3, 2, 2)

        surf_recorder = _LatentRecorder()
        lev_recorder = _LatentRecorder()
        post = PostPipeline(
            states_surface=surf_recorder,
            states_levels=lev_recorder,
        )
        pre = PrePipeline(
            states_surface=IdentityPreModule(),
            states_levels=IdentityPreModule(),
        )

        (
            input_tensor,
            latent_surface,
            latent_levels,
        ) = preprocess_data(
            states_surface=states_surface,
            states_levels=states_levels,
            pre_pipeline=pre,
            post_pipeline=post,
        )

        exp_surf_target = states_surface[:, -1]
        exp_surf_initial = states_surface[:, -2]
        exp_lev_target = states_levels[:, -1]
        exp_lev_initial = states_levels[:, -2]

        rec_surf_t, rec_surf_i = (
            surf_recorder.calls[0]
        )
        rec_lev_t, rec_lev_i = (
            lev_recorder.calls[0]
        )

        torch.testing.assert_close(
            rec_surf_t, exp_surf_target,
        )
        torch.testing.assert_close(
            rec_surf_i, exp_surf_initial,
        )
        torch.testing.assert_close(
            rec_lev_t, exp_lev_target,
        )
        torch.testing.assert_close(
            rec_lev_i, exp_lev_initial,
        )
        torch.testing.assert_close(
            latent_surface,
            exp_surf_target - exp_surf_initial,
        )
        torch.testing.assert_close(
            latent_levels,
            exp_lev_target - exp_lev_initial,
        )
        assert (
            input_tensor.shape[0]
            == states_surface.shape[0]
        )

    def test_split_wd_params_splits(self) -> None:
        r'''Weight-decay split separates param types.'''
        model = _ProbeNet()
        wd_params, nowd_params = split_wd_params(
            model,
        )

        wd_ids = {id(p) for p in wd_params}
        nowd_ids = {id(p) for p in nowd_params}

        assert id(model.linear.weight) in wd_ids
        assert id(model.linear.bias) in nowd_ids
        assert id(model.norm.weight) in nowd_ids
        assert (
            id(model.embedding.weight) in nowd_ids
        )
        assert id(model.frozen) not in wd_ids
        assert id(model.frozen) not in nowd_ids

        assert wd_ids.isdisjoint(nowd_ids)
        trainable_ids = {
            id(p)
            for p in model.parameters()
            if p.requires_grad
        }
        assert wd_ids | nowd_ids == trainable_ids


# -----------------------------------------------------------
# Unit tests
# -----------------------------------------------------------

class TestModulesUnittest:
    r'''Isolated unit tests for module internals.'''

    def test_on_train_batch_end_ema(self) -> None:
        r'''EMA network updates after train batch end.'''
        module = _build_training_module(
            ema_rate=0.0,
        )
        with torch.no_grad():
            module.network.linear.weight.fill_(2.0)

        module.on_train_batch_end()

        ema_weight = (
            module.ema_network.module.linear.weight
        )
        torch.testing.assert_close(
            ema_weight,
            module.network.linear.weight,
        )

    def test_on_train_end_bn(
            self,
            monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        r'''on_train_end delegates to SWA BN update.'''
        module = _build_training_module(
            ema_rate=0.0,
        )

        class _TrainerStub:
            def __init__(self) -> None:
                self.train_dataloader = object()

        module.trainer = _TrainerStub()
        recorded: dict = {}

        def _fake_update_bn(
                train_dataloader: object,
                ema_network: object,
        ) -> None:
            recorded["train_dataloader"] = (
                train_dataloader
            )
            recorded["ema_network"] = ema_network

        monkeypatch.setattr(
            torch.optim.swa_utils,
            "update_bn",
            _fake_update_bn,
        )

        module.on_train_end()

        assert (
            recorded["train_dataloader"]
            is module.trainer.train_dataloader
        )
        assert (
            recorded["ema_network"]
            is module.ema_network
        )

    def test_base_auxiliary_noop(self) -> None:
        r'''Validation works with base no-op aux loss.'''
        module = _BaseAuxTrainingModule(
            network=_ProbeNet(),
            pre_pipeline=PrePipeline(
                states_surface=_ParamPreModule(),
            ),
            post_pipeline=PostPipeline(
                states_surface=_ParamPostModule(),
            ),
            ema_rate=0.0,
        )

        out = module.validation_step(
            batch={}, batch_idx=0,
        )

        assert "loss" in out


# -----------------------------------------------------------
# Error tests
# -----------------------------------------------------------

class TestForecastModuleErrors:
    r'''Error condition tests for ForecastModule.'''

    def test_forward_raises_not_implemented(
            self,
    ) -> None:
        r'''Base forward raises NotImplementedError.'''
        module = _build_forecast_module()
        with pytest.raises(NotImplementedError):
            ForecastModule.forward(module)


class TestModulesErrors:
    r'''Error condition tests for modules package.'''

    def test_test_and_predict_raise(self) -> None:
        r'''Unsupported test/predict raise ValueError.'''
        module = _build_training_module()
        with pytest.raises(ValueError):
            module.test_step(
                batch={}, batch_idx=0,
            )
        with pytest.raises(ValueError):
            module.predict_step(
                batch={}, batch_idx=0,
            )

    def test_split_wd_intersecting(self) -> None:
        r'''Raises when name in multiple target sets.'''
        model = _IntersectingSetsModel()

        with pytest.raises(
            AssertionError, match="different sets",
        ):
            split_wd_params(model)

    def test_split_wd_missing(self) -> None:
        r'''Raises when parameter keys missing union.'''
        model = _MissingKeysModel()

        with pytest.raises(
            (AssertionError, TypeError),
            match=(
                "not separated"
                "|unsupported format string"
            ),
        ):
            split_wd_params(model)


# -----------------------------------------------------------
# Compiled network state_dict tests
# -----------------------------------------------------------

class TestTrainingModuleCompiledKeys:
    r'''Tests for state_dict key normalization when network
    is compiled with torch.compile.'''

    def test_state_dict_clean_when_network_compiled(
            self,
    ) -> None:
        r'''Compiled network state_dict has no _orig_mod keys.'''
        # Arrange
        module = _build_training_module()
        module.network = torch.compile(
            module.network, backend="eager",
        )

        # Act
        state = module.state_dict()

        # Assert
        for key in state:
            assert "._orig_mod" not in key, (
                f"Key {key!r} contains ._orig_mod"
            )

    def test_state_dict_clean_when_network_not_compiled(
            self,
    ) -> None:
        r'''Uncompiled network state_dict has no _orig_mod keys.'''
        # Arrange
        module = _build_training_module()

        # Act
        state = module.state_dict()

        # Assert
        for key in state:
            assert "._orig_mod" not in key, (
                f"Key {key!r} contains ._orig_mod"
            )

    def test_state_dict_ema_keys_clean_when_compiled(
            self,
    ) -> None:
        r'''EMA keys have no _orig_mod after compile.'''
        # Arrange
        module = _build_training_module(ema_rate=0.0)
        module.network = torch.compile(
            module.network, backend="eager",
        )

        # Act
        state = module.state_dict()

        # Assert
        ema_keys = [
            k for k in state
            if k.startswith("ema_network.module.")
        ]
        assert len(ema_keys) > 0, (
            "Expected EMA keys in state_dict"
        )
        for key in ema_keys:
            assert "._orig_mod" not in key, (
                f"EMA key {key!r} contains ._orig_mod"
            )
