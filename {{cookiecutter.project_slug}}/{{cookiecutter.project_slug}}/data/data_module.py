#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules
import logging
import os.path
from typing import Optional, Iterable

# External modules
import lightning.pytorch as pl
import torch.cuda
from torch.utils.data import DataLoader

# Internal modules
from .dataset import TrainDataset

__all__ = [
    "TrainDataModule",
]


main_logger = logging.getLogger(__name__)


class TrainDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for training data stored in zarr format.

    Provides train and validation dataloaders for the surrogate model training
    pipeline.
    """
    def __init__(
            self,
            data_path: str,
            state_variables: Iterable[str],
            forcing_variables: Optional[Iterable[str]] = None,
            auxiliary_path: Optional[str] = None,
            auxiliary_variables: Optional[Iterable[str]] = None,
            n_steps: int = 2,
            n_step_size: int = 1,
            batch_size: int = 64,
            val_batch_size: int = 16,
            n_workers: int = 4,
            pin_memory: bool = True,
    ):
        r'''
        Initialize the DataModule.

        Parameters
        ----------
        data_path : str
            Base path to the dataset directory containing `train.zarr` and
            `val.zarr`.
        state_variables : Iterable[str]
            List of state variable names to include in the samples.
        forcing_variables : Iterable[str], optional, default None
            List of forcing variable names to include in the samples. If None,
            no forcing variables are included.
        auxiliary_path : str, optional, default None
            Path to an auxiliary netCDF dataset containing static variables
            (e.g. mesh, land-sea mask). If None, no auxiliary data is included.
        auxiliary_variables : Iterable[str], optional, default None
            List of variable names to retrieve from the auxiliary dataset. Only
            used if auxiliary_path is not None.
        n_steps : int, default 2
            Number of time steps to include in each sample.
        n_step_size : int, default 1
            Step size between consecutive time steps in each sample.
        batch_size : int, default 64
            Batch size for training.
        val_batch_size : int, default 16
            Batch size for validation.
        n_workers : int, default 4
            Number of data loading workers.
        pin_memory : bool, default True
            Whether to pin memory for faster GPU transfer, suitable for Nvidia
            GPUs devices. Is ignored if CUDA is not available.
        '''
        super().__init__()
        self._train_dataset: Optional[TrainDataset] = None
        self._val_dataset: Optional[TrainDataset] = None
        self.data_path = data_path
        self.state_variables = state_variables
        self.forcing_variables = forcing_variables
        self.auxiliary_path = auxiliary_path
        self.auxiliary_variables = auxiliary_variables
        self.n_steps = n_steps
        self.n_step_size = n_step_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        """
        Set up datasets for the given stage.

        Parameters
        ----------
        stage : str
            One of 'fit', 'validate', or 'predict'.
        """
        if stage == "fit":
            self._train_dataset = TrainDataset(
                os.path.join(self.data_path, "train.zarr"),
                state_variables=self.state_variables,
                forcing_variables=self.forcing_variables,
                auxiliary_path=self.auxiliary_path,
                auxiliary_variables=self.auxiliary_variables,
                n_steps=self.n_steps,
                n_step_size=self.n_step_size
            )
        if stage in ("fit", "validate"):
            self._val_dataset = TrainDataset(
                os.path.join(self.data_path, "val.zarr"),
                state_variables=self.state_variables,
                forcing_variables=self.forcing_variables,
                auxiliary_path=self.auxiliary_path,
                auxiliary_variables=self.auxiliary_variables,
                n_steps=self.n_steps,
                n_step_size=self.n_step_size
            )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self._train_dataset, batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            num_workers=self.n_workers,
            persistent_workers=self.n_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            num_workers=self.n_workers,
            persistent_workers=self.n_workers > 0,
        )
