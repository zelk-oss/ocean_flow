ocean_flow.modules package
===================

The :py:mod:`ocean_flow.modules` package contains high-level training and forecasting
module abstractions built on PyTorch Lightning. It connects network forward
passes with pre/post pipelines and centralizes shared orchestration logic.

Overview
--------

- :py:mod:`ocean_flow.modules.forecast_module` provides a pure abstract base class
   for single-step forecast modules.  Subclasses implement
   :py:meth:`~ocean_flow.modules.forecast_module.ForecastModule.forward`;
   prediction-loop and zarr I/O logic lives outside this class.
- :py:mod:`ocean_flow.modules.train_module` provides an abstract training module with
   common Lightning hooks, EMA handling, and optimizer/scheduler setup.
- :py:mod:`ocean_flow.modules.utils` provides helper functions for input processing,
   latent target preparation, and parameter-group construction.

Submodules
----------

ocean_flow.modules.forecast\_module module
-----------------------------------

.. automodule:: ocean_flow.modules.forecast_module
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.modules.train\_module module
--------------------------------

.. automodule:: ocean_flow.modules.train_module
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.modules.utils module
------------------------

.. automodule:: ocean_flow.modules.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: ocean_flow.modules
   :members:
   :undoc-members:
   :show-inheritance:
