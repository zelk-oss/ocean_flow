{{cookiecutter.project_slug}}.pipelines package
=====================

The :py:mod:`{{cookiecutter.project_slug}}.pipelines` package provides composable preprocessing and
postprocessing layers for variable transforms between physical space and latent
network space.

Overview
--------

This package defines:

- base interfaces for pre-processing
   (:py:class:`{{cookiecutter.project_slug}}.pipelines.pre_module.PreModule`) and post-processing
   (:py:class:`{{cookiecutter.project_slug}}.pipelines.post_module.PostModule`),
- sequential composition containers
   (:py:class:`{{cookiecutter.project_slug}}.pipelines.PrePipeline`,
   :py:class:`{{cookiecutter.project_slug}}.pipelines.PostPipeline`), and
- concrete implementations for normalization, tendency conversion, and
   lower-bound enforcement.

These components are used by training and forecasting modules to keep data
transforms explicit, configurable, and testable.

Base modules
------------

{{cookiecutter.project_slug}}.pipelines.pre\_module module
--------------------------------

.. automodule:: {{cookiecutter.project_slug}}.pipelines.pre_module
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.pipelines.post\_module module
---------------------------------

.. automodule:: {{cookiecutter.project_slug}}.pipelines.post_module
   :members:
   :undoc-members:
   :show-inheritance:

Implementations
---------------

{{cookiecutter.project_slug}}.pipelines.bounding module
-----------------------------

.. automodule:: {{cookiecutter.project_slug}}.pipelines.bounding
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.pipelines.normalization module
----------------------------------

.. automodule:: {{cookiecutter.project_slug}}.pipelines.normalization
   :members:
   :undoc-members:
   :show-inheritance:

{{cookiecutter.project_slug}}.pipelines.pipelines module
------------------------------

.. automodule:: {{cookiecutter.project_slug}}.pipelines.pipelines
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

{{cookiecutter.project_slug}}.pipelines.tendency module
-----------------------------

.. automodule:: {{cookiecutter.project_slug}}.pipelines.tendency
   :members:
   :undoc-members:
   :show-inheritance:

Other utilities
---------------

{{cookiecutter.project_slug}}.pipelines.utils module
--------------------------

.. automodule:: {{cookiecutter.project_slug}}.pipelines.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: {{cookiecutter.project_slug}}.pipelines
   :members:
   :undoc-members:
   :show-inheritance:
