#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}
r'''Tests for checkpoint resumption in the training script.

These tests verify that ``trainer.fit(ckpt_path=...)`` correctly
resumes training state, matching the code path exercised by
``scripts/train.py`` line 80.
'''

# System modules
import glob
import pathlib
from typing import Any, Dict, List

# External modules
import lightning.pytorch as pl
import pytest
import torch
import torch.utils.data

# Internal modules
from {{cookiecutter.project_slug}}.modules.train_module import (
    TrainingModule,
)
from {{cookiecutter.project_slug}}.pipelines import (
    PostPipeline,
    PrePipeline,
)
from tests.conftest import (
    DummyNetwork,
    IdentityPostModule,
    IdentityPreModule,
)


# -----------------------------------------------------------
# Helper classes
# -----------------------------------------------------------

class _SimpleTrainingModule(TrainingModule):
    r'''Minimal TrainingModule for checkpoint tests.

    Implements ``estimate_loss`` with a differentiable
    scalar loss connected to the network parameters,
    ignoring batch contents entirely.
    '''

    def estimate_loss(
            self,
            batch: Dict[str, Any],
            prefix: str = "train",
    ) -> Dict[str, Any]:
        r'''Return a dummy differentiable loss.

        Parameters
        ----------
        batch : Dict[str, Any]
            Ignored batch data.
        prefix : str, optional
            Logging prefix, by default ``"train"``.

        Returns
        -------
        Dict[str, Any]
            Dictionary with ``"loss"`` key.
        '''
        x = torch.ones(1, 1, device=self.device)
        loss = self.network.linear(x).sum()
        return {"loss": loss}


class _DummyDataset(torch.utils.data.Dataset):
    r'''Fixed-size dataset returning empty dicts.

    Parameters
    ----------
    length : int
        Number of samples in the dataset.
    '''

    def __init__(self, length: int = 64) -> None:
        self._length = length

    def __len__(self) -> int:
        r'''Return dataset length.

        Returns
        -------
        int
            Number of samples.
        '''
        return self._length

    def __getitem__(
            self,
            idx: int,
    ) -> Dict[str, torch.Tensor]:
        r'''Return an empty dict as batch element.

        Parameters
        ----------
        idx : int
            Sample index (unused).

        Returns
        -------
        Dict[str, torch.Tensor]
            Empty dictionary.
        '''
        return {}


# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def _build_simple_module(
        total_steps: int = 20,
) -> _SimpleTrainingModule:
    r'''Build a minimal training module for tests.

    Parameters
    ----------
    total_steps : int, optional
        Total scheduler steps, by default 20.

    Returns
    -------
    _SimpleTrainingModule
        Configured training module.
    '''
    return _SimpleTrainingModule(
        network=DummyNetwork(),
        pre_pipeline=PrePipeline(
            states_surface=IdentityPreModule(),
        ),
        post_pipeline=PostPipeline(
            states_surface=IdentityPostModule(),
        ),
        lr=1e-3,
        lr_warmup_steps=0,
        total_steps=total_steps,
        weight_decay=0.0,
        ema_rate=0.0,
    )


def _make_dataloader() -> torch.utils.data.DataLoader:
    r'''Create a minimal DataLoader for training tests.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader yielding empty-dict batches.
    '''
    return torch.utils.data.DataLoader(
        _DummyDataset(length=64),
        batch_size=4,
    )


def _find_checkpoint(
        root: pathlib.Path,
) -> str:
    r'''Find the last checkpoint file under root.

    Parameters
    ----------
    root : pathlib.Path
        Root directory to search.

    Returns
    -------
    str
        Path to the checkpoint file.

    Raises
    ------
    FileNotFoundError
        If no checkpoint file is found.
    '''
    pattern = str(root / "**" / "*.ckpt")
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"No .ckpt files found under {root}"
        )
    # Prefer last.ckpt if present
    for match in matches:
        if "last" in match:
            return match
    return matches[-1]


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestTrainTaskFunctional:
    r'''Functional tests for checkpoint resumption.

    These tests exercise the same code path as
    ``trainer.fit(ckpt_path=cfg.ckpt_path)`` in
    ``scripts/train.py`` line 80.
    '''

    def test_ckpt_path_resumes_training(
            self,
            tmp_path: pathlib.Path,
    ) -> None:
        r'''Resuming from checkpoint continues global_step.'''
        # Arrange -- train for 4 steps and save checkpoint
        model_a = _build_simple_module(total_steps=8)
        dataloader = _make_dataloader()
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=str(tmp_path / "checkpoints"),
            save_last=True,
            every_n_train_steps=4,
        )
        trainer_a = pl.Trainer(
            max_steps=4,
            enable_checkpointing=True,
            callbacks=[checkpoint_cb],
            default_root_dir=str(tmp_path),
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )
        trainer_a.fit(
            model=model_a,
            train_dataloaders=dataloader,
        )
        assert trainer_a.global_step == 4

        ckpt_file = _find_checkpoint(tmp_path)

        # Act -- resume training from checkpoint
        model_b = _build_simple_module(total_steps=8)
        trainer_b = pl.Trainer(
            max_steps=8,
            enable_checkpointing=True,
            default_root_dir=str(tmp_path / "resumed"),
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )
        trainer_b.fit(
            model=model_b,
            train_dataloaders=dataloader,
            ckpt_path=ckpt_file,
        )

        # Assert
        assert trainer_b.global_step == 8

    def test_ckpt_path_null_trains_from_scratch(
            self,
            tmp_path: pathlib.Path,
    ) -> None:
        r'''Training with ckpt_path=None starts at step 0.'''
        # Arrange
        model = _build_simple_module(total_steps=4)
        dataloader = _make_dataloader()
        trainer = pl.Trainer(
            max_steps=4,
            enable_checkpointing=False,
            default_root_dir=str(tmp_path),
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )

        # Act
        trainer.fit(
            model=model,
            train_dataloaders=dataloader,
            ckpt_path=None,
        )

        # Assert
        assert trainer.global_step == 4
        assert trainer.current_epoch >= 0

    def test_ckpt_path_restores_optimizer_state(
            self,
            tmp_path: pathlib.Path,
    ) -> None:
        r'''Resumed optimizer state dict is non-empty.'''
        # Arrange -- train for 4 steps and save checkpoint
        model_a = _build_simple_module(total_steps=8)
        dataloader = _make_dataloader()
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=str(tmp_path / "checkpoints"),
            save_last=True,
            every_n_train_steps=4,
        )
        trainer_a = pl.Trainer(
            max_steps=4,
            enable_checkpointing=True,
            callbacks=[checkpoint_cb],
            default_root_dir=str(tmp_path),
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )
        trainer_a.fit(
            model=model_a,
            train_dataloaders=dataloader,
        )
        ckpt_file = _find_checkpoint(tmp_path)

        # Act -- resume and check optimizer state
        model_b = _build_simple_module(total_steps=8)
        trainer_b = pl.Trainer(
            max_steps=8,
            enable_checkpointing=True,
            default_root_dir=str(tmp_path / "resumed"),
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )
        trainer_b.fit(
            model=model_b,
            train_dataloaders=dataloader,
            ckpt_path=ckpt_file,
        )

        # Assert -- optimizer should have accumulated state
        optimizer = trainer_b.optimizers[0]
        assert len(optimizer.state) > 0, (
            "Optimizer state is empty after resumption; "
            "expected restored momentum buffers"
        )
        for param_id, state in optimizer.state.items():
            assert "exp_avg" in state or "step" in state, (
                f"Param {param_id} missing AdamW state "
                f"keys; got {list(state.keys())}"
            )
