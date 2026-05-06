Autoregressive Ensemble Forecasting
===================================

This guide explains how to run the autoregressive ensemble forecast pipeline
provided by :py:mod:`{{cookiecutter.project_slug}}.forecast`. The pipeline performs multi-step
rollouts over an ensemble of members using a two-phase architecture:
a **global phase** validates inputs and creates the output store in the
main process, then a **local phase** runs inference on each Fabric worker
with its own Dask client.

Overview
--------

The forecast pipeline splits execution into two distinct phases.

**Global phase** (main process, before ``fabric.launch()``):

1. Validation functions from
   :py:mod:`~{{cookiecutter.project_slug}}.forecast.validation` check that all input
   stores, checkpoint files, and dask addresses are valid.
2. :py:func:`~{{cookiecutter.project_slug}}.forecast.create_output_store` creates (or
   :py:func:`~{{cookiecutter.project_slug}}.forecast.validate_output_store` validates) the
   output zarr store.
3. :py:func:`~{{cookiecutter.project_slug}}.forecast.generate_forecast_configs` produces
   batched :py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastConfig` objects with
   data-parallel-aware batch sizing.

**Local phase** (each Fabric worker independently):

1. :py:func:`~{{cookiecutter.project_slug}}.forecast.setup_environment` configures logging
   and sets a per-worker random seed.
2. :py:func:`~{{cookiecutter.project_slug}}.forecast.initialize_client` opens a per-worker
   Dask client (selecting from a scheduler list by DP rank if configured).
3. An :py:class:`~{{cookiecutter.project_slug}}.forecast.InputReader` loads zarr/netCDF
   inputs via persist-based prefetching.
4. :py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastModel` advances the model
   over ``lead_times`` under Lightning Fabric.
5. An :py:class:`~{{cookiecutter.project_slug}}.forecast.OutputWriter` returns
   ``dask.delayed`` objects that write predictions directly to zarr regions.
6. After ``dask.compute()`` and ``fabric.barrier()``, rank 0 checks output
   completeness.

Ensemble members and initialisation times are batched through
:py:func:`~{{cookiecutter.project_slug}}.forecast.generate_forecast_configs`, which produces
independent :py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastConfig` objects.

Data
----

Training and prediction use the same variable families
(``state_variables``, ``forcing_variables``, ``auxiliary_variables``), but
they are sampled differently.

Training data structure
~~~~~~~~~~~~~~~~~~~~~~~

- Training/validation inputs come from zarr stores (typically
  ``train.zarr`` and ``val.zarr``) with a ``time`` axis.
- Each training sample contains state and forcing windows of shape
  ``(n_times, ...)``.
- Optional auxiliary variables are loaded from a netCDF file and added as
  static fields for every sample.

Prediction data structure
~~~~~~~~~~~~~~~~~~~~~~~~~

- Forecast initial states are read from zarr for the requested
  ``(init_time, ensemble)`` pairs, yielding arrays shaped ``(batch, ...)``.
- Optional forcings are read from zarr across requested lead times, yielding
  ``(batch, lead_time, ...)``.
- Optional auxiliary fields are loaded from netCDF for forecasting, then
  indexed by ensemble to produce ``(batch, ...)`` arrays.
- Auxiliary loading is activated only when both ``auxiliary_path`` and
  ``auxiliary_variables`` are configured.

State vs forcing vs auxiliary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **State data**: model state variables that are advanced autoregressively and
  written as outputs.
- **Forcing data**: known time-varying inputs used during rollout but not
  directly written as forecast targets.
- **Auxiliary data**: static context fields (for example masks/mesh geometry)
  loaded once from netCDF and reused at each forecast step.

Auxiliary ensemble behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- If auxiliary data has no ``ensemble`` dimension, a singleton ensemble axis
  is added and reused for all requested members.
- If auxiliary data includes ``ensemble``, requested ensemble members are
  mapped to available entries (with wrap-around when needed).

Prediction output structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- After prediction, only state variables are written to the output zarr store.
- Output arrays are indexed by ``init_time``, ``lead_time``, and
  ``ensemble``, plus the original spatial/state dimensions.
- Unwritten forecast regions remain ``NaN`` until the corresponding write is
  executed.

Configuration
-------------

:py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastConfig` is a dataclass that holds all
parameters required for a single forecast batch:

- ``init_times`` – ``pd.DatetimeIndex`` of initialisation datetimes for
  this batch.
- ``lead_times`` – ``pd.TimedeltaIndex`` of all lead times to produce.
- ``ens_mems`` – integer array of ensemble member indices for this batch.
- ``n_store_freq`` – size of each lead-time write chunk; controls how
  iteration over the config yields sub-sequences of ``lead_times``.

Iterating over a :py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastConfig` (``for chunk in
config``) yields successive ``pd.TimedeltaIndex`` slices of length
``n_store_freq``.  This is the mechanism by which the forecast runner
subdivides a long forecast into sequential write operations.

``batch_size`` is **not** a :py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastConfig`
field -- it is consumed only by
:py:func:`~{{cookiecutter.project_slug}}.forecast.generate_forecast_configs` to control how many
``(init_time, ens_mem)`` pairs are grouped into each config object.
When running with data parallelism, the ``dp_world_size`` parameter divides
``batch_size`` to produce a per-worker batch size
(``max(1, batch_size // dp_world_size)``).

:py:func:`~{{cookiecutter.project_slug}}.forecast.generate_forecast_configs` accepts a Hydra
``DictConfig`` with the keys: ``init_start``, ``init_end``, ``init_freq``
(forwarded to :func:`pandas.date_range`); ``lead_time`` and ``step_freq``
(forwarded to :func:`pandas.timedelta_range`); plus ``ensemble_size``,
``batch_size``, and ``n_store_freq``. An optional ``dp_world_size``
parameter (default ``1``) controls per-worker batch sizing.

.. code-block:: python

   import pandas as pd
   from omegaconf import OmegaConf
   from {{cookiecutter.project_slug}}.forecast import generate_forecast_configs

   cfg = OmegaConf.create({
       "init_start": "2020-01-01T00:00",
       "init_end": "2020-01-03T00:00",
       "init_freq": "1D",
       "lead_time": "240h",
       "step_freq": "6h",
       "ensemble_size": 3,
       "batch_size": 3,
       "n_store_freq": 2,
   })

   for config in generate_forecast_configs(cfg):
       print(config.init_times, config.ens_mems)

Multi-Step Input and Output
---------------------------

The forecast stack supports multi-step input windows and multi-step prediction
chunks. :py:class:`~{{cookiecutter.project_slug}}.forecast.InputReader` loads historical state slices
ending at each initialisation time, and
:py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastModel` advances the autoregressive window by
appending each module output and retaining the trailing input context.

Input windows
~~~~~~~~~~~~~

``n_in_steps`` defines how many consecutive state time steps are loaded for
each forecast initialisation. The configured ``init_time`` is the final entry
of that window. If ``n_in_steps`` is greater than one, ``step_freq`` defines
the spacing between the loaded states.

- ``load_states`` returns arrays shaped ``(batch, n_in_steps, ...)``.
- For ``n_in_steps=2`` and ``step_freq="6h"``, the loaded states correspond
  to ``[T-6h, T]`` for an initialisation time ``T``.
- ``step_freq`` is required when ``n_in_steps > 1`` because the historical
  offsets are derived from :py:func:`pandas.Timedelta`.

Forcing windows
~~~~~~~~~~~~~~~

The forcing window contains both the historical context and the future forcing
steps needed for the rollout. :py:class:`~{{cookiecutter.project_slug}}.forecast.InputReader` therefore
loads forcing arrays with shape ``(batch, n_in_steps + len(lead_times), ...)``.

- The first ``n_in_steps`` entries cover the historical window ending at
  ``init_time``.
- The remaining entries correspond to ``init_time + lead_times``.
- For ``n_in_steps=2`` and three requested lead times, the forcing time axis
  has length five.

Prediction chunks
~~~~~~~~~~~~~~~~~

``n_out_steps`` defines how many forecast steps a module emits per forward
call. :py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastModel` validates that every returned
tensor has this time-axis length, updates the rolling state window, and
concatenates the prediction chunks into the final trajectory.

- ``advance(n)`` performs ``n`` autoregressive calls, not ``n`` single-step
  updates.
- The returned forecast arrays therefore have shape
  ``(batch, n * n_out_steps, ...)``.
- The internal state always retains the trailing ``n_in_steps`` entries after
  each call.

Moving-window example
~~~~~~~~~~~~~~~~~~~~~

For :py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastModel` the state is advanced using a
shifting window over the time axis. With ``n_in_steps=2`` and
``n_out_steps=2``, an initialisation at time ``T`` uses state ``[T-6h, T]``.
The first ``advance(n=1)`` call yields predictions at ``[T+6h, T+12h]`` and
updates the internal state to ``[T+6h, T+12h]`` (dropping ``T-6h`` and ``T``).
The second call would predict ``[T+18h, T+24h]`` and update state to
``[T+18h, T+24h]``.

For forcing variables, the window includes historical context plus requested
future lead times. With ``n_in_steps=2`` and ``lead_times=[6h, 12h, 18h, 24h]``
the loaded forcing axis is:

- historical: ``[T-6h, T]``
- future: ``[T+6h, T+12h, T+18h, T+24h]``

Each module advance consumes a moving slice of length ``n_in_steps + n_out_steps``.
The first call uses ``[T-6h .. T+12h]``, the second call uses ``[T+6h .. T+24h]``.
This is consistent with chunked rollout of multiple lead times, where the
full lead set is produced by repeated calls to ``advance``.

.. code-block:: yaml

   n_in_steps: 2
   n_out_steps: 2
   step_freq: "6h"

.. note::

   ``n_store_freq`` should be divisible by ``n_out_steps`` so each write chunk
   aligns with the model output chunks. The framework does not validate this
   constraint.

I/O: reading inputs and writing outputs
---------------------------------------

InputReader
~~~~~~~~~~~

:py:class:`~{{cookiecutter.project_slug}}.forecast.InputReader` opens three categories of dataset:

- **State** – a zarr store (``data_path``) containing the initial model
  state variables.
- **Forcing** – an optional zarr store supplying time-varying external inputs
  that are not forecast targets.
- **Auxiliary** – optional static fields loaded from netCDF (e.g.
  land–sea mask, grid geometry).

The private method :py:meth:`~{{cookiecutter.project_slug}}.forecast.InputReader._open_optional_dataset`
provides a unified interface for datasets that may or may not be present: it
returns ``None`` when optional variables are not configured; for forcings, it
can use either a dedicated zarr path or the default state dataset.

OutputWriter
~~~~~~~~~~~~

:py:class:`~{{cookiecutter.project_slug}}.forecast.OutputWriter` writes forecast predictions
directly to a pre-created zarr output store using ``dask.delayed`` region
writes. The store has dimensions
``(init_time, lead_time, ensemble, ...)`` where trailing dimensions follow
each state variable's spatial layout in the reference store.

- Store creation and validation are handled separately by
  :py:func:`~{{cookiecutter.project_slug}}.forecast.create_output_store` and
  :py:func:`~{{cookiecutter.project_slug}}.forecast.validate_output_store` in the global
  phase, before any workers are launched.
- During initialisation, the writer opens the reference zarr store with
  xarray to extract spatial metadata, then discards the xarray dataset.
  No xarray objects are retained after ``__init__`` completes.
- The output store is opened via ``zarr.open(store_path, mode='r+')``
  for all subsequent writes.
- Each call to
  :py:meth:`~{{cookiecutter.project_slug}}.forecast.OutputWriter.write` returns a list of
  ``dask.delayed`` objects. Each delayed object writes a numpy array
  directly into the correct zarr region.
- The caller collects delayed objects and executes them with
  ``dask.compute()`` after all forecast batches have been processed.

Forecast runner and Dask IO
---------------------------

:py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastModel` wraps a
:py:class:`~{{cookiecutter.project_slug}}.modules.ForecastModule` and runs entirely on each
Fabric worker under Lightning Fabric. This decoupling from Dask enables
multi-GPU strategies such as FSDP and DDP.

Device placement and mixed-precision autocast are delegated entirely to a
:py:class:`lightning.fabric.Fabric` instance stored as
``ForecastModel.fabric``. The module setter calls ``fabric.setup(module)``
to handle both single-GPU and multi-GPU distribution transparently.
``fabric.autocast()`` provides the precision context for each forward
pass.

When running with data parallelism,
:py:func:`~{{cookiecutter.project_slug}}.forecast.run_forecast` accepts ``dp_rank`` and
``dp_world_size`` parameters. Each worker processes a strided subset of
the forecast configs: ``configs[dp_rank::dp_world_size]``. This avoids
overlap between workers without explicit coordination.

All IO -- reading states, auxiliary data, and forcings, as well as
writing trajectories -- is handled through persist-based prefetching
and ``dask.delayed`` writes. The
:py:func:`~{{cookiecutter.project_slug}}.forecast.run_forecast` function orchestrates the
outer loop over forecast configurations, while
:py:func:`~{{cookiecutter.project_slug}}.forecast.run_batch` handles the inner loop over
lead-time chunks for a single batch. After all delayed writes are
collected, ``dask.compute()`` executes them.

Prefetching
~~~~~~~~~~~

Two prefetch depth parameters control how far ahead IO tasks are
submitted relative to the current inference step.

- ``n_prefetch_init`` (default ``1``) governs how many additional
  configurations' states and auxiliary data are persisted before
  the runner blocks on the current batch.
- ``n_prefetch_forcing`` (default ``3``) governs how many forcing
  chunks are persisted ahead of the current lead-time chunk.

These are configured in the ``dask:`` section of
``configs/forecast.yaml``:

.. code-block:: yaml

   dask:
     scheduler: null  # null (local), string (shared), or list (per-worker)
     n_workers: 1
     dashboard_address: null
     n_prefetch_init: 1
     n_prefetch_forcing: 3

Higher values increase memory consumption on Dask workers but reduce
idle time between inference steps. The defaults are suitable for
most workloads.

.. note::

   The ``dask.processes`` configuration option has been removed.
   When ``dask.scheduler`` is ``null``, a thread-only
   ``distributed.LocalCluster`` is always created with
   ``processes=False``. This avoids conflicts with Fabric's
   multiprocessing.

Write execution
~~~~~~~~~~~~~~~

:py:class:`~{{cookiecutter.project_slug}}.forecast.OutputWriter` returns ``dask.delayed``
objects from each call to
:py:meth:`~{{cookiecutter.project_slug}}.forecast.OutputWriter.write`. These are collected
across all forecast batches. After the forecast loop completes,
:py:func:`~{{cookiecutter.project_slug}}.forecast.run_forecast` calls
``dask.compute(*all_delayed)`` to execute all writes in a single pass.
This ensures writes are batched efficiently rather than executed
individually.

Conceptual data-flow
~~~~~~~~~~~~~~~~~~~~

Each Fabric worker executes the following sequence independently.
When running with data parallelism, each worker processes a disjoint
subset of the forecast configs (strided by DP rank).

::

   Per-worker (Fabric)            Dask (IO, per worker)
   ===================            =====================
   run_forecast                   persist(load_states)
     |                            persist(load_auxiliary)
     +-- run_batch                persist(load_forcings)
     |     set_state(result)
     |     set_auxiliary(result)
     |     advance(n=...)
     |     write -> List[Delayed]
     |
     +-- dask.compute(*delayed)
     +-- fabric.barrier()
     +-- rank 0: completeness check

The :py:class:`~{{cookiecutter.project_slug}}.forecast.runner.PrefetchIterator` class
provides a reusable deque-based iterator for persisting work ahead of
consumption. It maintains a bounded buffer of in-flight items and
yields results in submission order.

.. autosummary::

   {{cookiecutter.project_slug}}.forecast.runner.run_forecast
   {{cookiecutter.project_slug}}.forecast.runner.run_batch
   {{cookiecutter.project_slug}}.forecast.runner.initialize_io
   {{cookiecutter.project_slug}}.forecast.runner.PrefetchIterator

Hardware and precision
----------------------

Device placement, parallelism strategy, and mixed-precision control are
configured through the ``trainer:`` section of ``configs/forecast.yaml``.
This section contains a subset of trainer keys used to configure
Lightning Fabric for inference. Unlike the full ``trainer:`` block in
``configs/train.yaml``, it omits training-specific fields such as
``max_epochs``.

The four keys and their defaults are:

.. code-block:: yaml

   trainer:
     accelerator: "cpu"      # 'cpu', 'gpu', or 'mps'
     devices: 1              # int or list of device indices
     precision: "32-true"    # '32-true', 'bf16-mixed', '16-mixed'
     strategy: "auto"        # 'auto', 'ddp', or strategy dict

The ``strategy`` key controls the distributed execution strategy.
Setting ``strategy: "ddp"`` with ``devices: 4`` launches four
data-parallel workers. Setting ``strategy: "auto"`` lets Fabric choose
the appropriate strategy based on the number of devices.

Lightning Fabric is a lightweight wrapper from the PyTorch Lightning project
that handles device placement, parallelism, and mixed-precision arithmetic
without requiring a full training loop setup. These values are consumed by
``_launch_forecast()`` in ``scripts/forecast.py`` to construct a
:py:class:`lightning.fabric.Fabric` instance. That instance is passed to
each worker via ``fabric.launch()``, and
:py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastModel` receives it for
device placement. The model's ``module.setter`` calls
``fabric.setup(module)`` once. Each forward pass in ``_step`` then runs
inside ``fabric.autocast()`` for precision control.

To run inference on a GPU with bfloat16 mixed precision, override the
``trainer`` keys from the command line:

.. code-block:: bash

   python scripts/forecast.py \
       trainer.accelerator=gpu \
       trainer.devices=1 \
       trainer.precision=bf16-mixed \
       ckpt_path=data/models/flow/my_model.ckpt \
       ...

A ``computer/`` profile (e.g. ``configs/computer/modern_multicuda.yaml``)
can also set these keys as a group so the same override applies to both
training and forecasting runs:

.. code-block:: yaml

   # configs/computer/modern_multicuda.yaml
   trainer:
     accelerator: "gpu"
     devices: [0, 1]
     precision: "bf16-mixed"
     strategy: "ddp"

To apply a computer profile, pass it with the ``+computer=`` flag. The
``+`` prefix instructs Hydra to merge a config group that is not listed
in the file defaults:

.. code-block:: bash

   python scripts/forecast.py +computer=modern_multicuda \
       ckpt_path=data/models/best.ckpt

This overrides the ``trainer:`` defaults with the GPU settings defined
in the chosen profile (``accelerator``, ``devices``, ``precision``,
``strategy``).

.. note::

   ``ForecastModel`` retains a separate ``dtype`` parameter (default
   ``torch.float32``) for state tensor storage. This is independent of the
   Fabric ``precision`` setting, which controls model weight and activation
   dtype. State tensors remain in full precision regardless of autocast.

API References
~~~~~~~~~~~~~~

.. autosummary::

   {{cookiecutter.project_slug}}.forecast.forecast_model.ForecastModel
   {{cookiecutter.project_slug}}.forecast.environment.setup_environment
   {{cookiecutter.project_slug}}.forecast.environment.initialize_client
   {{cookiecutter.project_slug}}.forecast.checkpoint.load_forecast_model

Running a forecast
------------------

The Hydra entry point is ``scripts/forecast.py``. Configuration is assembled
from files under ``configs/`` (``data/``, ``forecast_module/``, ``computer/``)
and can be overridden on the command line:

.. code-block:: bash

   python scripts/forecast.py \
       forecast_module=generative_flow \
       +computer=macbook \
       ckpt_path=data/models/flow/my_model.ckpt \
       io.data_path=data/train_data/wb2_64x32 \
       init_start='2020-01-01' \
       init_end='2020-01-07' \
       init_freq='1D' \
       lead_time='240h' \
       step_freq='6h' \
       ensemble_size=3 \
       batch_size=4 \
       n_store_freq=2

Key config groups:

- ``data/`` (e.g. ``wb2_64x32.yaml``) -- dataset paths and variable lists.
- ``forecast_module/`` (e.g. ``generative_flow.yaml``) -- module class and
  checkpoint path.
- ``computer/`` (e.g. ``macbook.yaml``) -- ``trainer`` hardware keys
  (accelerator, devices, precision, strategy).
- ``trainer:`` inline block -- device, precision, and strategy defaults (see
  `Hardware and precision`_ above).

Global validation
-----------------

Before any Fabric workers are launched, the global phase validates all
inputs and configuration. This ensures that errors such as missing
stores, absent variables, or invalid dask addresses are caught early
in a single process rather than manifesting as confusing failures
across multiple workers.

The :py:mod:`~{{cookiecutter.project_slug}}.forecast.validation` module provides seven
pure-function validators. All operate on paths and arrays without
requiring class state.

Store and checkpoint validators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~{{cookiecutter.project_slug}}.forecast.validate_initial_conditions` opens the
initial-condition zarr store and verifies that every configured state
variable exists. It raises ``FileNotFoundError`` if the store is absent
and ``ValueError`` if a variable is missing.

:py:func:`~{{cookiecutter.project_slug}}.forecast.validate_auxiliary` checks the optional
auxiliary netCDF path and variables. Both must be set or both must be
``None``; a mismatch raises ``ValueError``.

:py:func:`~{{cookiecutter.project_slug}}.forecast.validate_forcing` validates forcing
variables against either a dedicated forcing zarr store or the default
data store.

:py:func:`~{{cookiecutter.project_slug}}.forecast.validate_checkpoint` verifies that the
model checkpoint file exists on disk.

Output store management
~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~{{cookiecutter.project_slug}}.forecast.create_output_store` creates a NaN-filled
zarr output store with correct dimensions, spatial metadata, and chunk
sizes (``1, n_store_freq, 1, *spatial``). It uses xarray for one-time
store creation in the global phase. When ``recreate=True``, any
existing store is deleted first.

:py:func:`~{{cookiecutter.project_slug}}.forecast.validate_output_store` verifies that an
existing output store has the expected shape for the configured
``init_times``, ``lead_times``, and ``ens_mems``. This is used during
restart runs to confirm the store is compatible.

Dask address validation
~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~{{cookiecutter.project_slug}}.forecast.validate_dask_addresses` checks that
when ``dask.scheduler`` is a list, its length matches the number of
data-parallel workers. A single string or ``None`` always passes
validation.

.. autosummary::

   {{cookiecutter.project_slug}}.forecast.validation.validate_initial_conditions
   {{cookiecutter.project_slug}}.forecast.validation.validate_auxiliary
   {{cookiecutter.project_slug}}.forecast.validation.validate_forcing
   {{cookiecutter.project_slug}}.forecast.validation.validate_checkpoint
   {{cookiecutter.project_slug}}.forecast.validation.validate_output_store
   {{cookiecutter.project_slug}}.forecast.validation.create_output_store
   {{cookiecutter.project_slug}}.forecast.validation.validate_dask_addresses

Distributed data-parallel forecasting
-------------------------------------

The pipeline supports distributed data parallel (DDP) execution through
Lightning Fabric. Multiple workers divide the forecast configs among
themselves and write to the same pre-created zarr output store.

Setting up multi-GPU execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run a forecast on four GPUs with DDP, configure the ``trainer:``
section accordingly.

.. code-block:: yaml

   trainer:
     accelerator: "gpu"
     devices: 4
     precision: "bf16-mixed"
     strategy: "ddp"

Fabric spawns one worker per device. Each worker receives a distinct
``global_rank`` used to compute its data-parallel rank (``dp_rank``).

Per-worker config striding
~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~{{cookiecutter.project_slug}}.forecast.run_forecast` distributes forecast
configs across workers using a stride pattern. With four workers and
twelve configs, worker ``dp_rank=k`` processes configs at indices
``[k, k+4, k+8]``. This guarantees each config is handled by exactly
one worker with no overlap.

Per-worker batch sizing
~~~~~~~~~~~~~~~~~~~~~~~

The ``batch_size`` in the configuration is the **global** batch size.
:py:func:`~{{cookiecutter.project_slug}}.forecast.generate_forecast_configs` divides it by
``dp_world_size`` to produce per-worker batches. With ``batch_size=8``
and four workers, each
:py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastConfig` contains two
``(init_time, ensemble_member)`` pairs.

Per-worker Dask clients
~~~~~~~~~~~~~~~~~~~~~~~

Each Fabric worker initialises its own Dask client via
:py:func:`~{{cookiecutter.project_slug}}.forecast.initialize_client`. The ``dask.scheduler``
config key supports three modes:

- ``null`` -- each worker creates a thread-only
  ``distributed.LocalCluster`` (``processes=False`` always).
- A single string -- all workers connect to the same scheduler.
- A list of strings -- worker ``dp_rank=k`` connects to
  ``scheduler[k]``.

Per-worker random seeds
~~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~{{cookiecutter.project_slug}}.forecast.setup_environment` accepts a
``worker_rank`` parameter. When a global ``seed`` is configured, the
actual seed is ``seed + worker_rank``. This ensures reproducible but
distinct random sequences across workers.

Synchronisation and completeness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After all delayed writes are computed, each worker calls
``fabric.barrier()`` to synchronise. Rank 0 then performs a
completeness check to verify that all output store cells have been
written (no ``NaN`` values remain).

.. autosummary::

   {{cookiecutter.project_slug}}.forecast.runner.run_forecast
   {{cookiecutter.project_slug}}.forecast.config.generate_forecast_configs
   {{cookiecutter.project_slug}}.forecast.environment.setup_environment
   {{cookiecutter.project_slug}}.forecast.environment.initialize_client
   {{cookiecutter.project_slug}}.forecast.validation.validate_dask_addresses
