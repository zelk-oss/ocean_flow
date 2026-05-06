ocean_flow.forecast package
==============================================

The :py:mod:`ocean_flow.forecast` package implements an autoregressive
ensemble forecast pipeline. The pipeline separates into two phases: a
**global phase** that validates inputs and creates the output store in the
main process, and a **local phase** where each Fabric worker runs inference
with its own Dask client.  `Dask distributed`_ handles all IO through
persist-based prefetching and ``dask.delayed`` region writes.

.. _Dask distributed: https://distributed.dask.org/en/stable/

Overview
--------

The package exports the following public symbols:

- :py:class:`~ocean_flow.forecast.ForecastConfig` and
  :py:func:`~ocean_flow.forecast.generate_forecast_configs` -- batch
  configuration management with data-parallel-aware batch sizing.
- :py:class:`~ocean_flow.forecast.ForecastModel` -- stateful
  autoregressive model wrapper.
- :py:class:`~ocean_flow.forecast.InputReader` and
  :py:class:`~ocean_flow.forecast.OutputWriter` -- zarr-based IO.
  :py:class:`~ocean_flow.forecast.OutputWriter` writes predictions
  directly to zarr via ``dask.delayed`` region writes without xarray in the
  write path.
- :py:func:`~ocean_flow.forecast.run_forecast` and
  :py:func:`~ocean_flow.forecast.run_batch` -- persist-based
  orchestration loop with prefetching and DP-rank-strided config iteration.
- :py:func:`~ocean_flow.forecast.initialize_io` -- IO factory from
  Hydra config.
- :py:func:`~ocean_flow.forecast.load_forecast_model` -- checkpoint
  loading and model construction.
- :py:func:`~ocean_flow.forecast.setup_environment` and
  :py:func:`~ocean_flow.forecast.initialize_client` -- runtime
  environment (with per-worker seeds) and Dask client setup (with per-worker
  scheduler selection).
- Seven global validation functions from
  :py:mod:`~ocean_flow.forecast.validation`:
  :py:func:`~ocean_flow.forecast.validate_initial_conditions`,
  :py:func:`~ocean_flow.forecast.validate_auxiliary`,
  :py:func:`~ocean_flow.forecast.validate_forcing`,
  :py:func:`~ocean_flow.forecast.validate_checkpoint`,
  :py:func:`~ocean_flow.forecast.validate_output_store`,
  :py:func:`~ocean_flow.forecast.create_output_store`, and
  :py:func:`~ocean_flow.forecast.validate_dask_addresses`.
  These run before ``fabric.launch()`` to catch configuration errors early.

Full documentation for each symbol is in the submodule sections below.

.. autosummary::

    ocean_flow.forecast.ForecastConfig
    ocean_flow.forecast.generate_forecast_configs
    ocean_flow.forecast.ForecastModel
    ocean_flow.forecast.InputReader
    ocean_flow.forecast.OutputWriter
    ocean_flow.forecast.run_forecast
    ocean_flow.forecast.run_batch
    ocean_flow.forecast.initialize_io
    ocean_flow.forecast.load_forecast_model
    ocean_flow.forecast.setup_environment
    ocean_flow.forecast.initialize_client
    ocean_flow.forecast.validate_initial_conditions
    ocean_flow.forecast.validate_auxiliary
    ocean_flow.forecast.validate_forcing
    ocean_flow.forecast.validate_checkpoint
    ocean_flow.forecast.validate_output_store
    ocean_flow.forecast.create_output_store
    ocean_flow.forecast.validate_dask_addresses

Submodules
----------

ocean_flow.forecast.validation module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ocean_flow.forecast.validation
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.forecast.config module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ocean_flow.forecast.config
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.forecast.forecast\_model module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ocean_flow.forecast.forecast_model
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.forecast.runner module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ocean_flow.forecast.runner
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.forecast.checkpoint module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ocean_flow.forecast.checkpoint
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.forecast.environment module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ocean_flow.forecast.environment
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.forecast.input module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ocean_flow.forecast.input
   :members:
   :private-members:
   :undoc-members:
   :show-inheritance:

ocean_flow.forecast.output module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ocean_flow.forecast.output
   :members:
   :undoc-members:
   :show-inheritance:
