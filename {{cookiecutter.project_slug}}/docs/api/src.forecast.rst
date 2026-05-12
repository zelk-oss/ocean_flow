{{cookiecutter.project_slug}}.forecast package
==============================================

The :py:mod:`{{cookiecutter.project_slug}}.forecast` package implements an autoregressive
ensemble forecast pipeline. The pipeline separates into two phases: a
**global phase** that validates inputs and creates the output store in the
main process, and a **local phase** where each Fabric worker runs inference
with its own Dask client.  `Dask distributed`_ handles all IO through
persist-based prefetching and ``dask.delayed`` region writes.

.. _Dask distributed: https://distributed.dask.org/en/stable/

Overview
--------

The package exports the following public symbols:

- :py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastConfig` and
  :py:func:`~{{cookiecutter.project_slug}}.forecast.generate_forecast_configs` -- batch
  configuration management with data-parallel-aware batch sizing.
- :py:class:`~{{cookiecutter.project_slug}}.forecast.ForecastModel` -- stateful
  autoregressive model wrapper.
- :py:class:`~{{cookiecutter.project_slug}}.forecast.InputReader` and
  :py:class:`~{{cookiecutter.project_slug}}.forecast.OutputWriter` -- zarr-based IO.
  :py:class:`~{{cookiecutter.project_slug}}.forecast.OutputWriter` writes predictions
  directly to zarr via ``dask.delayed`` region writes without xarray in the
  write path.
- :py:func:`~{{cookiecutter.project_slug}}.forecast.run_forecast` and
  :py:func:`~{{cookiecutter.project_slug}}.forecast.run_batch` -- persist-based
  orchestration loop with prefetching and DP-rank-strided config iteration.
- :py:func:`~{{cookiecutter.project_slug}}.forecast.initialize_io` -- IO factory from
  Hydra config.
- :py:func:`~{{cookiecutter.project_slug}}.forecast.load_forecast_model` -- checkpoint
  loading and model construction.
- :py:func:`~{{cookiecutter.project_slug}}.forecast.setup_environment` and
  :py:func:`~{{cookiecutter.project_slug}}.forecast.initialize_client` -- runtime
  environment (with per-worker seeds) and Dask client setup (with per-worker
  scheduler selection).
- Seven global validation functions from
  :py:mod:`~{{cookiecutter.project_slug}}.forecast.validation`:
  :py:func:`~{{cookiecutter.project_slug}}.forecast.validate_initial_conditions`,
  :py:func:`~{{cookiecutter.project_slug}}.forecast.validate_auxiliary`,
  :py:func:`~{{cookiecutter.project_slug}}.forecast.validate_forcing`,
  :py:func:`~{{cookiecutter.project_slug}}.forecast.validate_checkpoint`,
  :py:func:`~{{cookiecutter.project_slug}}.forecast.validate_output_store`,
  :py:func:`~{{cookiecutter.project_slug}}.forecast.create_output_store`, and
  :py:func:`~{{cookiecutter.project_slug}}.forecast.validate_dask_addresses`.
  These run before ``fabric.launch()`` to catch configuration errors early.

Full documentation for each symbol is in the submodule sections below.

.. autosummary::

    {{cookiecutter.project_slug}}.forecast.ForecastConfig
    {{cookiecutter.project_slug}}.forecast.generate_forecast_configs
    {{cookiecutter.project_slug}}.forecast.ForecastModel
    {{cookiecutter.project_slug}}.forecast.InputReader
    {{cookiecutter.project_slug}}.forecast.OutputWriter
    {{cookiecutter.project_slug}}.forecast.run_forecast
    {{cookiecutter.project_slug}}.forecast.run_batch
    {{cookiecutter.project_slug}}.forecast.initialize_io
    {{cookiecutter.project_slug}}.forecast.load_forecast_model
    {{cookiecutter.project_slug}}.forecast.setup_environment
    {{cookiecutter.project_slug}}.forecast.initialize_client
    {{cookiecutter.project_slug}}.forecast.validate_initial_conditions
    {{cookiecutter.project_slug}}.forecast.validate_auxiliary
    {{cookiecutter.project_slug}}.forecast.validate_forcing
    {{cookiecutter.project_slug}}.forecast.validate_checkpoint
    {{cookiecutter.project_slug}}.forecast.validate_output_store
    {{cookiecutter.project_slug}}.forecast.create_output_store
    {{cookiecutter.project_slug}}.forecast.validate_dask_addresses

Submodules
----------

{{cookiecutter.project_slug}}.forecast.validation module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: {{cookiecutter.project_slug}}.forecast.validation
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.forecast.config module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: {{cookiecutter.project_slug}}.forecast.config
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.forecast.forecast\_model module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: {{cookiecutter.project_slug}}.forecast.forecast_model
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.forecast.runner module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: {{cookiecutter.project_slug}}.forecast.runner
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.forecast.checkpoint module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: {{cookiecutter.project_slug}}.forecast.checkpoint
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.forecast.environment module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: {{cookiecutter.project_slug}}.forecast.environment
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.forecast.input module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: {{cookiecutter.project_slug}}.forecast.input
   :members:
   :private-members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.forecast.output module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: {{cookiecutter.project_slug}}.forecast.output
   :members:
   :undoc-members:
   :show-inheritance:
