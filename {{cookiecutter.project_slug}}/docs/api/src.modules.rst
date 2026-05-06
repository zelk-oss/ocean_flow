{{cookiecutter.project_slug}}.modules package
===================

The :py:mod:`{{cookiecutter.project_slug}}.modules` package contains high-level training and forecasting
module abstractions built on PyTorch Lightning. It connects network forward
passes with pre/post pipelines and centralizes shared orchestration logic.

Overview
--------

- :py:mod:`{{cookiecutter.project_slug}}.modules.forecast_module` provides a pure abstract base class
   for single-step forecast modules.  Subclasses implement
   :py:meth:`~{{cookiecutter.project_slug}}.modules.forecast_module.ForecastModule.forward`;
   prediction-loop and zarr I/O logic lives outside this class.
- :py:mod:`{{cookiecutter.project_slug}}.modules.train_module` provides an abstract training module with
   common Lightning hooks, EMA handling, and optimizer/scheduler setup.
- :py:mod:`{{cookiecutter.project_slug}}.modules.utils` provides helper functions for input processing,
   latent target preparation, and parameter-group construction.

Submodules
----------

{{cookiecutter.project_slug}}.modules.forecast\_module module
-----------------------------------

.. automodule:: {{cookiecutter.project_slug}}.modules.forecast_module
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.modules.train\_module module
--------------------------------

.. automodule:: {{cookiecutter.project_slug}}.modules.train_module
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.modules.utils module
------------------------

.. automodule:: {{cookiecutter.project_slug}}.modules.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: {{cookiecutter.project_slug}}.modules
   :members:
   :undoc-members:
   :show-inheritance:
