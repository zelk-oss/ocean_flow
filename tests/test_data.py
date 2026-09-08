# -*- coding: utf-8 -*-
r'''Tests for ocean_flow/modules/data.py – FlowMatchingTrainDataModule.'''

# External modules
import numpy as np
import pytest
from torch.utils.data import Dataset

# Internal modules
import flow_train.data.data_module as data_module_mod
from ocean_flow.modules.data import (
    FlowMatchingTrainDataModule,
    _WithTrainingSample,
    assemble_training_sample,
)


# -----------------------------------------------------------
# Helper classes
# -----------------------------------------------------------

class _StubDataset(Dataset):
    r'''Minimal dataset returning fixed raw samples.'''

    def __init__(self, samples: list) -> None:
        self._samples = samples

    def __len__(self) -> int:
        r'''Return number of stored samples.'''
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        r'''Return the sample dict at idx, copied.'''
        return dict(self._samples[idx])


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestAssembleTrainingSampleFunctional:
    r'''Functional tests for assemble_training_sample.'''

    def test_adds_input_and_residual_for_scalar_variable(self) -> None:
        r'''Scalar (2D per step) variables get a channel axis inserted.'''
        state_series = np.stack([
            np.full((4, 8), 1.0, dtype=np.float32),
            np.full((4, 8), 3.0, dtype=np.float32),
        ])
        sample = {"q": state_series}

        out = assemble_training_sample(sample, ["q"])

        assert out["input"].shape == (1, 4, 8)
        assert out["residual"].shape == (1, 4, 8)
        np.testing.assert_allclose(out["input"], 1.0)
        np.testing.assert_allclose(out["residual"], 2.0)

    def test_concatenates_multiple_state_variables(self) -> None:
        r'''Multiple variables are concatenated along the channel axis.'''
        q_series = np.stack([
            np.full((1, 4, 8), 1.0, dtype=np.float32),
            np.full((1, 4, 8), 2.0, dtype=np.float32),
        ])
        p_series = np.stack([
            np.full((1, 4, 8), 5.0, dtype=np.float32),
            np.full((1, 4, 8), 5.0, dtype=np.float32),
        ])
        sample = {"q": q_series, "p": p_series}

        out = assemble_training_sample(sample, ["q", "p"])

        assert out["input"].shape == (2, 4, 8)
        assert out["residual"].shape == (2, 4, 8)

    def test_noop_when_no_state_variables(self) -> None:
        r'''sample is returned unchanged when state_variables is empty.'''
        sample = {"q": np.zeros((2, 4, 8))}

        out = assemble_training_sample(sample, [])

        assert "input" not in out
        assert "residual" not in out


class TestWithTrainingSampleFunctional:
    r'''Functional tests for the _WithTrainingSample dataset wrapper.'''

    def test_len_matches_wrapped_dataset(self) -> None:
        r'''__len__ delegates to the wrapped dataset.'''
        inner = _StubDataset([{"q": np.zeros((2, 1, 4, 8))}] * 3)

        wrapped = _WithTrainingSample(inner, ["q"])

        assert len(wrapped) == 3

    def test_getitem_adds_training_keys(self) -> None:
        r'''__getitem__ assembles input/residual on top of raw samples.'''
        q_series = np.stack([
            np.full((1, 4, 8), 1.0, dtype=np.float32),
            np.full((1, 4, 8), 4.0, dtype=np.float32),
        ])
        inner = _StubDataset([{"q": q_series}])

        wrapped = _WithTrainingSample(inner, ["q"])
        sample = wrapped[0]

        np.testing.assert_allclose(sample["input"], 1.0)
        np.testing.assert_allclose(sample["residual"], 3.0)


class TestFlowMatchingTrainDataModuleFunctional:
    r'''Functional tests for FlowMatchingTrainDataModule.setup.'''

    def _build_module(self) -> FlowMatchingTrainDataModule:
        r'''Return a FlowMatchingTrainDataModule with dummy config.'''
        return FlowMatchingTrainDataModule(
            data_path="/tmp/does-not-matter",
            state_variables=["q"],
        )

    def test_setup_fit_wraps_train_and_val_datasets(
            self,
            monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        r'''setup("fit") wraps both train and val datasets.'''
        monkeypatch.setattr(
            data_module_mod, "TrainDataset", _StubDataset,
        )
        monkeypatch.setattr(
            _StubDataset, "__init__",
            lambda self, *args, **kwargs: setattr(
                self, "_samples", [],
            ),
            raising=False,
        )
        module = self._build_module()

        module.setup("fit")

        assert isinstance(module._train_dataset, _WithTrainingSample)
        assert isinstance(module._val_dataset, _WithTrainingSample)

    def test_setup_validate_wraps_only_val_dataset(
            self,
            monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        r'''setup("validate") wraps only the val dataset.'''
        monkeypatch.setattr(
            data_module_mod, "TrainDataset", _StubDataset,
        )
        monkeypatch.setattr(
            _StubDataset, "__init__",
            lambda self, *args, **kwargs: setattr(
                self, "_samples", [],
            ),
            raising=False,
        )
        module = self._build_module()

        module.setup("validate")

        assert module._train_dataset is None
        assert isinstance(module._val_dataset, _WithTrainingSample)

    def test_setup_predict_wraps_neither_dataset(
            self,
            monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        r'''setup("predict") leaves both datasets unset.'''
        monkeypatch.setattr(
            data_module_mod, "TrainDataset", _StubDataset,
        )
        module = self._build_module()

        module.setup("predict")

        assert module._train_dataset is None
        assert module._val_dataset is None
