#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

# System modules
import logging
from typing import Dict, List, Optional

# External modules
import numpy as np
from torch.utils.data import Dataset

# Internal modules
from flow_train.data import TrainDataModule


main_logger = logging.getLogger(__name__)


def assemble_training_sample(
        sample: Dict[str, np.ndarray],
        state_variables: List[str],
) -> Dict[str, np.ndarray]:
    r'''
    Convert raw state samples to training inputs and residual targets.

    The flow-matching training module expects each batch to contain
    ``input`` and ``residual`` keys. ``input`` is the state at the
    first time step, and ``residual`` is the change to the last time
    step.

    Parameters
    ----------
    sample : Dict[str, np.ndarray]
        Raw sample as produced by ``TrainDataset.__getitem__``, keyed
        by state-variable name.
    state_variables : List[str]
        Names of the state variables to assemble into ``input`` and
        ``residual``.

    Returns
    -------
    Dict[str, np.ndarray]
        ``sample`` with ``input`` and ``residual`` keys added, if
        ``state_variables`` is non-empty.
    '''
    if state_variables:
        input_parts = []
        residual_parts = []
        for state_var in state_variables:
            state_series = sample[state_var]
            state_t = state_series[0]
            state_tdt = state_series[-1]
            if state_t.ndim == 2:
                state_t = state_t[None, ...]
                state_tdt = state_tdt[None, ...]
            input_parts.append(state_t)
            residual_parts.append(state_tdt - state_t)
        sample["input"] = np.concatenate(input_parts, axis=0)
        sample["residual"] = np.concatenate(residual_parts, axis=0)
    return sample


class _WithTrainingSample(Dataset):
    r'''
    Dataset wrapper adding ``input``/``residual`` keys to each sample.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The wrapped ``TrainDataset`` instance.
    state_variables : List[str]
        Names of the state variables to assemble into ``input`` and
        ``residual``.
    '''

    def __init__(
            self,
            dataset: Dataset,
            state_variables: List[str],
    ) -> None:
        self._dataset = dataset
        self._state_variables = state_variables

    def __len__(self) -> int:
        r'''Return the number of samples in the wrapped dataset.'''
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        r'''Return the wrapped sample with ``input``/``residual`` added.'''
        sample = self._dataset[idx]
        return assemble_training_sample(sample, self._state_variables)


class FlowMatchingTrainDataModule(TrainDataModule):
    r'''
    ``TrainDataModule`` that assembles flow-matching training samples.

    Wraps the train/validation ``TrainDataset`` instances so each
    sample additionally carries ``input``/``residual`` keys, as
    required by :class:`~ocean_flow.modules.training.FlowMatchingTrainingModule`.
    '''

    def setup(self, stage: str) -> None:
        r'''
        Set up datasets for the given stage, wrapping them for training.

        Parameters
        ----------
        stage : str
            One of 'fit', 'validate', or 'predict'.
        '''
        super().setup(stage)
        if self._train_dataset is not None:
            self._train_dataset = _WithTrainingSample(
                self._train_dataset, self.state_variables,
            )
        if self._val_dataset is not None:
            self._val_dataset = _WithTrainingSample(
                self._val_dataset, self.state_variables,
            )
