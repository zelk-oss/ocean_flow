ocean_flow.pipelines package
=====================

The :py:mod:`ocean_flow.pipelines` package provides composable preprocessing and
postprocessing layers for variable transforms between physical space and latent
network space.

Overview
--------

This package defines:

- base interfaces for pre-processing
   (:py:class:`ocean_flow.pipelines.pre_module.PreModule`) and post-processing
   (:py:class:`ocean_flow.pipelines.post_module.PostModule`),
- sequential composition containers
   (:py:class:`ocean_flow.pipelines.PrePipeline`,
   :py:class:`ocean_flow.pipelines.PostPipeline`), and
- concrete implementations for normalization, tendency conversion, and
   lower-bound enforcement.

These components are used by training and forecasting modules to keep data
transforms explicit, configurable, and testable.

Base modules
------------

ocean_flow.pipelines.pre\_module module
--------------------------------

.. automodule:: ocean_flow.pipelines.pre_module
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.pipelines.post\_module module
---------------------------------

.. automodule:: ocean_flow.pipelines.post_module
   :members:
   :undoc-members:
   :show-inheritance:

Implementations
---------------

ocean_flow.pipelines.bounding module
-----------------------------

.. automodule:: ocean_flow.pipelines.bounding
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.pipelines.normalization module
----------------------------------

.. automodule:: ocean_flow.pipelines.normalization
   :members:
   :undoc-members:
   :show-inheritance:

ocean_flow.pipelines.pipelines module
------------------------------

.. automodule:: ocean_flow.pipelines.pipelines
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ocean_flow.pipelines.tendency module
-----------------------------

.. automodule:: ocean_flow.pipelines.tendency
   :members:
   :undoc-members:
   :show-inheritance:

Other utilities
---------------

ocean_flow.pipelines.utils module
--------------------------

.. automodule:: ocean_flow.pipelines.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: ocean_flow.pipelines
   :members:
   :undoc-members:
   :show-inheritance:
