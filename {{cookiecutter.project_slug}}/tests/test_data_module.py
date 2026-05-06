# -*- coding: utf-8 -*-
r'''
Tests for the TrainDataModule class in src/data/data_module.py.

The tests aim for 100% coverage of data_module.py.
'''

# External modules
import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler as _sampler
import pytest

# Internal modules
from {{cookiecutter.project_slug}}.data.dataset import TrainDataset
from {{cookiecutter.project_slug}}.data.data_module import TrainDataModule
from tests.conftest import _create_test_zarr


class TestDataModuleFunctional:
    r'''End-to-end tests for TrainDataModule.'''

    def test_setup_fit_creates_both_datasets(
        self, tmp_path, monkeypatch,
    ):
        r'''setup('fit') creates both train and val datasets.'''
        base = tmp_path / "data"
        base.mkdir()
        _create_test_zarr(str(base / "train.zarr"), n_times=4)
        _create_test_zarr(str(base / "val.zarr"), n_times=4)
        monkeypatch.setattr(
            torch.cuda, "is_available", lambda: False,
        )
        tdm = TrainDataModule(
            data_path=str(base),
            state_variables=["states_surface"],
            forcing_variables=["states_levels"],
            batch_size=2,
            val_batch_size=1,
            n_workers=0,
            pin_memory=True,
        )
        tdm.setup("fit")
        assert isinstance(tdm._train_dataset, TrainDataset)
        assert isinstance(tdm._val_dataset, TrainDataset)

    def test_setup_fit_dataloaders(
        self, tmp_path, monkeypatch,
    ):
        r'''train/val dataloaders work after setup('fit').'''
        base = tmp_path / "data"
        base.mkdir()
        _create_test_zarr(str(base / "train.zarr"), n_times=4)
        _create_test_zarr(str(base / "val.zarr"), n_times=4)
        monkeypatch.setattr(
            torch.cuda, "is_available", lambda: False,
        )
        tdm = TrainDataModule(
            data_path=str(base),
            state_variables=["states_surface"],
            batch_size=2,
            val_batch_size=1,
            n_workers=0,
            pin_memory=True,
        )
        tdm.setup("fit")
        train_dl = tdm.train_dataloader()
        val_dl = tdm.val_dataloader()
        assert isinstance(train_dl, DataLoader)
        assert isinstance(val_dl, DataLoader)
        assert train_dl.batch_size == 2
        assert val_dl.batch_size == 1
        assert isinstance(
            train_dl.sampler, _sampler.RandomSampler,
        )
        assert isinstance(
            val_dl.sampler, _sampler.SequentialSampler,
        )
        assert train_dl.pin_memory is False
        assert val_dl.pin_memory is False
        _ = next(iter(train_dl))
        _ = next(iter(val_dl))

    def test_with_auxiliary_data(
        self, tmp_path, mocked_auxiliary_netcdf,
    ):
        r'''DataModule passes auxiliary config to datasets.'''
        base = tmp_path / "data"
        base.mkdir()
        _create_test_zarr(str(base / "train.zarr"), n_times=4)
        _create_test_zarr(str(base / "val.zarr"), n_times=4)
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=4, n_lon=8,
        )
        tdm = TrainDataModule(
            data_path=str(base),
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mask", "mesh"],
            n_workers=0,
        )
        tdm.setup("fit")
        assert (
            tdm._train_dataset.auxiliary_variables
            == ["mask", "mesh"]
        )
        assert (
            tdm._val_dataset.auxiliary_variables
            == ["mask", "mesh"]
        )


class TestDataModuleUnittest:
    r'''Isolated unit tests for TrainDataModule.'''

    def test_default_attributes(self, tmp_path):
        r'''DataModule default attribute values are correct.'''
        tdm = TrainDataModule(
            data_path=str(tmp_path),
            state_variables=["states_surface"],
        )
        assert tdm._train_dataset is None
        assert tdm._val_dataset is None
        assert tdm.batch_size == 64
        assert tdm.val_batch_size == 16
        assert tdm.n_workers == 4
        assert tdm.pin_memory is True
        assert tdm.n_steps == 2
        assert tdm.n_step_size == 1

    def test_pin_memory_with_cuda(
        self, tmp_path, monkeypatch,
    ):
        r'''pin_memory honoured when CUDA is available.'''
        base = tmp_path / "data"
        base.mkdir()
        _create_test_zarr(str(base / "train.zarr"), n_times=4)
        _create_test_zarr(str(base / "val.zarr"), n_times=4)
        monkeypatch.setattr(
            torch.cuda, "is_available", lambda: True,
        )
        tdm = TrainDataModule(
            data_path=str(base),
            state_variables=["states_surface"],
            batch_size=1,
            val_batch_size=1,
            n_workers=0,
            pin_memory=True,
        )
        tdm.setup("fit")
        assert tdm.train_dataloader().pin_memory is True
        assert tdm.val_dataloader().pin_memory is True

    def test_no_pin_memory(self, tmp_path, monkeypatch):
        r'''pin_memory=False results in no pinning.'''
        base = tmp_path / "data"
        base.mkdir()
        _create_test_zarr(str(base / "train.zarr"), n_times=4)
        _create_test_zarr(str(base / "val.zarr"), n_times=4)
        monkeypatch.setattr(
            torch.cuda, "is_available", lambda: True,
        )
        tdm = TrainDataModule(
            data_path=str(base),
            state_variables=["states_surface"],
            batch_size=1,
            val_batch_size=1,
            n_workers=0,
            pin_memory=False,
        )
        tdm.setup("fit")
        assert (
            tdm.train_dataloader().pin_memory is False
        )
        assert (
            tdm.val_dataloader().pin_memory is False
        )

    def test_setup_validate_only(self, tmp_path):
        r'''setup('validate') creates only val dataset.'''
        base = tmp_path / "data"
        base.mkdir()
        _create_test_zarr(str(base / "train.zarr"), n_times=4)
        _create_test_zarr(str(base / "val.zarr"), n_times=4)
        tdm = TrainDataModule(
            data_path=str(base),
            state_variables=["states_surface"],
        )
        tdm.setup("validate")
        assert tdm._train_dataset is None
        assert isinstance(tdm._val_dataset, TrainDataset)


class TestDataModuleEdgeCases:
    r'''Boundary condition tests for TrainDataModule.'''

    def test_setup_predict_no_datasets(self, tmp_path):
        r'''setup('predict') does not create any datasets.'''
        base = tmp_path / "data"
        base.mkdir()
        _create_test_zarr(
            str(base / "train.zarr"), n_times=2,
        )
        _create_test_zarr(
            str(base / "val.zarr"), n_times=2,
        )
        tdm = TrainDataModule(
            data_path=str(base),
            state_variables=["states_surface"],
        )
        tdm.setup("predict")
        assert tdm._train_dataset is None
        assert tdm._val_dataset is None
