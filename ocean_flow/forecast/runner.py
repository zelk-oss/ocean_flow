#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

r'''Forecast runner with persist-based prefetching.

Provides the main forecast loop with persist-based
prefetching of states, auxiliary data, and forcings via
lazy xarray Datasets backed by Dask arrays.
'''

from __future__ import annotations

# System modules
import collections
import logging
import math
from typing import (
    Callable,
    Generic,
    Iterator,
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    TypeVar,
)

# External modules
import dask
import distributed
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig
from tqdm.auto import tqdm

# Internal modules
from .config import _total_init_ens_pairs
from .input import InputReader, dataset_to_numpy_dict
from .output import OutputWriter

if TYPE_CHECKING:
    from .config import ForecastConfig
    from .forecast_model import ForecastModel


main_logger = logging.getLogger(__name__)

T = TypeVar("T")


__all__ = [
    "PrefetchIterator",
    "run_forecast",
    "run_batch",
    "initialize_io",
]


class PrefetchIterator(Generic[T]):
    r'''
    Iterator that prefetches items via ``.persist()``.

    Wraps a list of zero-argument callables. Each callable
    returns a lazy object supporting ``.persist()``. The
    iterator eagerly fills a lookahead buffer of size
    ``n_prefetch + 1`` and yields computed items while
    scheduling the next loads.

    Parameters
    ----------
    load_fns : list of callables
        Each callable takes no arguments and returns an
        object with a ``.persist()`` method (e.g. a lazy
        ``xr.Dataset``).
    n_prefetch : int, optional
        Number of items to prefetch ahead of consumption.
        The total buffer size is ``n_prefetch + 1``.
        Default is 1.
    '''

    def __init__(
            self,
            load_fns: List[Callable[[], T]],
            n_prefetch: int = 1,
    ) -> None:
        self._load_fns = list(load_fns)
        self._n_prefetch = max(n_prefetch, 0)
        self._buffer: collections.deque = (
            collections.deque()
        )
        self._next_idx = 0
        self._fill_buffer()

    def _fill_buffer(self) -> None:
        r'''Pre-fill the buffer up to capacity.'''
        capacity = min(
            self._n_prefetch + 1, len(self._load_fns),
        )
        while (
            self._next_idx < len(self._load_fns)
            and len(self._buffer) < capacity
        ):
            item = self._load_fns[self._next_idx]()
            if hasattr(item, "persist"):
                item = item.persist()
            self._buffer.append(item)
            self._next_idx += 1

    def __len__(self) -> int:
        r'''Return the total number of items.'''
        return len(self._load_fns)

    def __iter__(self) -> Iterator[T]:
        r'''Return self as iterator.'''
        return self

    def __next__(self) -> T:
        r'''
        Return the next prefetched item.

        Raises
        ------
        StopIteration
            When all items have been consumed.
        '''
        if not self._buffer:
            raise StopIteration
        item = self._buffer.popleft()
        if self._next_idx < len(self._load_fns):
            next_item = self._load_fns[self._next_idx]()
            if hasattr(next_item, "persist"):
                next_item = next_item.persist()
            self._buffer.append(next_item)
            self._next_idx += 1
        return item


def initialize_io(
        cfg: DictConfig,
) -> Tuple[InputReader, OutputWriter]:
    r'''
    Initialize the input reader and output writer.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing io, init, lead
        time, and ensemble settings.

    Returns
    -------
    input_reader : InputReader
        Reader for initial conditions, auxiliary, and
        forcings.
    output_writer : OutputWriter
        Writer for forecast trajectories.
    '''
    input_reader = InputReader(
        data_path=cfg.io.data_path,
        state_variables=cfg.io.state_variables,
        auxiliary_path=cfg.io.auxiliary_path,
        auxiliary_variables=cfg.io.auxiliary_variables,
        forcing_path=cfg.io.forcing_path,
        forcing_variables=cfg.io.forcing_variables,
        n_in_steps=cfg.get("n_in_steps", 1),
        step_freq=cfg.get("step_freq", None),
    )
    
    init_times = pd.date_range(
        start=cfg.init_start,
        end=cfg.init_end,
        freq=cfg.init_freq,
    )
    lead_times = pd.timedelta_range(
        start=cfg.step_freq,
        end=cfg.lead_time,
        freq=cfg.step_freq,
    )

    ensemble_members = np.arange(cfg.ensemble_size)
    output_writer = OutputWriter(
        data_path=cfg.io.data_path,
        state_variables=cfg.io.state_variables,
        store_path=cfg.io.store_path,
        init_times=init_times,
        lead_times=lead_times,
        ens_mems=ensemble_members,
    )
    return input_reader, output_writer


def _count_forecast_batches(
        cfg: DictConfig,
) -> int:
    r'''
    Return the total number of forecast batches.

    Parameters
    ----------
    cfg : DictConfig
        Hydra forecast configuration.

    Returns
    -------
    int
        Number of batches.
    '''
    init_start = cfg.get("init_start")
    init_end = cfg.get("init_end")
    init_freq = cfg.get("init_freq")
    if init_start is None or init_end is None or (
        init_freq is None
    ):
        return 1
    n_total = _total_init_ens_pairs(cfg)
    batch_size = cfg.get("batch_size", 1)
    return math.ceil(n_total / batch_size)


def _shift_forcing_times(
        config: ForecastConfig,
        chunk: pd.TimedeltaIndex,
        offset: pd.Timedelta,
) -> Tuple[pd.DatetimeIndex, pd.TimedeltaIndex]:
    r'''
    Shift init times and lead times for forcing loading.

    When processing chunks beyond the first, the init
    times must be shifted forward by the cumulative offset
    and the lead times adjusted relative to that offset.

    Parameters
    ----------
    config : ForecastConfig
        Forecast configuration.
    chunk : pd.TimedeltaIndex
        Lead-time chunk for forcing.
    offset : pd.Timedelta
        Cumulative offset from previous chunks.

    Returns
    -------
    shifted_init_times : DatetimeIndex
        Init times shifted by offset.
    relative_lead_times : TimedeltaIndex
        Lead times relative to offset.
    '''
    if offset == pd.Timedelta(0):
        return config.init_times, chunk
    shifted = config.init_times + offset
    relative = chunk - offset
    return shifted, relative


def run_forecast(
        client: distributed.Client,
        model: ForecastModel,
        input_reader: InputReader,
        output_writer: OutputWriter,
        forecast_configs: List[ForecastConfig],
        n_prefetch_init: int = 1,
        n_prefetch_forcing: int = 3,
        dp_rank: int = 0,
        dp_world_size: int = 1,
) -> None:
    r'''
    Run the forecast loop with persist-based prefetch.

    Uses :class:`PrefetchIterator` to overlap IO with
    inference. States and auxiliary data are prefetched
    ahead of consumption. All Delayed write objects are
    batch-computed at the end. When running with data
    parallelism, each worker processes a strided subset
    of the configs based on ``dp_rank`` and
    ``dp_world_size``.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    model : ForecastModel
        Forecast model for inference.
    input_reader : InputReader
        Input reader for states, auxiliary, forcings.
    output_writer : OutputWriter
        Output writer for trajectories.
    forecast_configs : iterable
        Iterable of forecast configurations.
    n_prefetch_init : int, optional
        Number of configs to prefetch ahead via
        .persist(). Default is 1.
    n_prefetch_forcing : int, optional
        Number of forcing chunks to prefetch. Default
        is 3.
    dp_rank : int, optional
        Data-parallel rank of this worker. Default is 0.
    dp_world_size : int, optional
        Total number of data-parallel workers. Default
        is 1.
    '''
    configs = list(forecast_configs)
    configs = configs[dp_rank::dp_world_size]
    if not configs:
        return

    def _load_init_data(cfg):
        states = input_reader.load_states(
            cfg.init_times, cfg.ens_mems,
        )
        aux = None
        if input_reader.use_auxiliary:
            aux = input_reader.load_auxiliary(
                cfg.ens_mems,
            )
        return states, aux

    load_fns = [
        lambda c=c: _load_init_data(c) for c in configs
    ]
    prefetch = PrefetchIterator(
        load_fns, n_prefetch=n_prefetch_init,
    )

    all_delayed: List = []
    progress_configs = tqdm(
        configs,
        total=len(configs),
        desc=f"rank {dp_rank}",
        position=dp_rank,
        disable=False,
    )
    for config, (states_ds, aux_ds) in zip(
        progress_configs, prefetch,
    ):
        batch_delayed = run_batch(
            model=model,
            input_reader=input_reader,
            output_writer=output_writer,
            config=config,
            states_ds=states_ds,
            aux_ds=aux_ds,
            n_prefetch_forcing=n_prefetch_forcing,
        )
        all_delayed.extend(batch_delayed)

    if all_delayed:
        dask.compute(*all_delayed)


def run_batch(
        model: ForecastModel,
        input_reader: InputReader,
        output_writer: OutputWriter,
        config: ForecastConfig,
        states_ds: xr.Dataset,
        aux_ds: Optional[xr.Dataset] = None,
        n_prefetch_forcing: int = 3,
) -> List:
    r'''
    Run a single forecast batch over lead-time chunks.

    Parameters
    ----------
    model : ForecastModel
        Forecast model for inference.
    input_reader : InputReader
        Input reader for forcings.
    output_writer : OutputWriter
        Output writer for trajectories.
    config : ForecastConfig
        Forecast configuration for this batch.
    states_ds : xr.Dataset
        Lazy or persisted states Dataset.
    aux_ds : xr.Dataset or None, optional
        Lazy or persisted auxiliary Dataset. Default is
        None.
    n_prefetch_forcing : int, optional
        Number of forcing chunks to prefetch. Default
        is 3.

    Returns
    -------
    list
        Delayed write objects for all chunks.
    '''
    chunks = list(config.get_leadtime_iterator())
    if not chunks:
        return []

    offsets: List[pd.Timedelta] = [pd.Timedelta(0)]
    for chunk in chunks[:-1]:
        offsets.append(chunk[-1])

    states_dict = dataset_to_numpy_dict(
        states_ds.compute(),
    )
    model.set_state(states_dict)

    if aux_ds is not None:
        aux_dict = dataset_to_numpy_dict(
            aux_ds.compute(),
        )
        model.set_auxiliary(aux_dict)

    forcing_iter = _build_forcing_prefetch(
        input_reader, config, chunks, offsets,
        n_prefetch_forcing,
    )

    write_delayed: List = []
    for chunk, forcings in zip(chunks, forcing_iter):
        if model.n_out_steps <= 0:
            raise ValueError(
                f"model.n_out_steps must be positive,"
                f" got {model.n_out_steps}"
            )

        n_requested = len(chunk)
        n_calls = math.ceil(
            n_requested / model.n_out_steps,
        )
        trajectory = model.advance(
            n=n_calls, forcings=forcings,
        )
        if n_requested > 0:
            for var, array in trajectory.items():
                if array.shape[1] > n_requested:
                    trajectory[var] = array[
                        :, :n_requested, ...
                    ]

        delayed = output_writer.write(
            trajectory,
            config.init_times,
            config.ens_mems,
            chunk,
        )
        write_delayed.extend(delayed)

    return write_delayed


def _build_forcing_prefetch(
        input_reader: InputReader,
        config: ForecastConfig,
        chunks: List[pd.TimedeltaIndex],
        offsets: List[pd.Timedelta],
        n_prefetch: int,
) -> Iterator:
    r'''
    Build a forcing prefetch iterator or a no-op iterator.

    Parameters
    ----------
    input_reader : InputReader
        Input reader for forcings.
    config : ForecastConfig
        Forecast configuration.
    chunks : list of pd.TimedeltaIndex
        Lead-time chunks.
    offsets : list of pd.Timedelta
        Cumulative offsets per chunk.
    n_prefetch : int
        Number of forcing chunks to prefetch.

    Returns
    -------
    iterator
        Yields forcing numpy dicts (or None for each
        chunk if forcings are disabled).
    '''
    if not input_reader.use_forcings:
        return iter([None] * len(chunks))

    def _load_forcing(j):
        shifted, relative = _shift_forcing_times(
            config, chunks[j], offsets[j],
        )
        return input_reader.load_forcings(
            shifted, config.ens_mems, relative,
        )

    load_fns = [
        lambda j=j: _load_forcing(j)
        for j in range(len(chunks))
    ]
    prefetch = PrefetchIterator(
        load_fns, n_prefetch=max(n_prefetch, 1) - 1,
    )
    return (
        dataset_to_numpy_dict(ds.compute())
        for ds in prefetch
    )
