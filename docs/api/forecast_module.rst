:orphan:

.. _api-forecast-module:

ForecastModule
==============

.. automodule:: ocean_flow.modules.forecast_module
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. rubric:: Overview

:py:class:`~ocean_flow.modules.forecast_module.ForecastModule` is an abstract
base class for single-step forecast models.  It stores a network and
two transformation pipelines, and declares :py:meth:`forward` as the
only contract that subclasses must satisfy.  All prediction-loop,
autoregressive-rollout, and zarr I/O logic lives **outside** this
class (see :doc:`/guides/forecasting` and ``scripts/forecast.py``).

Subclasses
----------

To create a concrete forecast module, inherit from
:py:class:`~ocean_flow.modules.forecast_module.ForecastModule` and implement
:py:meth:`~ocean_flow.modules.forecast_module.ForecastModule.forward`.

The example below shows the minimal pattern: pre-process the input
window with ``self.pre_pipeline``, run ``self.network``, and
post-process the output with ``self.post_pipeline``.

.. code-block:: python

   import torch
   from ocean_flow.modules.forecast_module import ForecastModule
   from ocean_flow.modules.utils import process_inputs


   class IncrementForecastModule(ForecastModule):
       r"""Minimal forecast module that adds a learned bias.

       Parameters
       ----------
       network : torch.nn.Module
           Neural network mapping the preprocessed input tensor
           to a flat output tensor.
       pre_pipeline : PrePipeline
           Input transformation pipeline.
       post_pipeline : PostPipeline
           Output transformation pipeline.
       """

       def forward(
               self,
               states_surface: torch.Tensor,
               states_levels: torch.Tensor,
       ) -> tuple[torch.Tensor, torch.Tensor]:
           r"""Predict the next surface and level states.

           Parameters
           ----------
           states_surface : torch.Tensor
               Surface state window with shape
               ``(batch, time_steps, n_vars, lat, lon)``.
           states_levels : torch.Tensor
               Level state window with shape
               ``(batch, time_steps, n_vars, levels, lat, lon)``.

           Returns
           -------
           pred_surface : torch.Tensor
               Predicted surface state with shape
               ``(batch, n_vars, lat, lon)``.
           pred_levels : torch.Tensor
               Predicted level state with shape
               ``(batch, n_vars, levels, lat, lon)``.
           """
           # Flatten time × variable channels and pre-process
           in_tensor = process_inputs(
               states_surface=states_surface,
               states_levels=states_levels,
               pre_pipeline=self.pre_pipeline,
           )

           # Forward pass through the network
           out_tensor = self.network(in_tensor)

           # Split and post-process surface output
           n_surf = states_surface.shape[2]
           raw_surface = out_tensor[:, :n_surf]
           pred_surface = self.post_pipeline["states_surface"](
               raw_surface, states_surface[:, -1],
           )

           # Pass through levels unchanged as a baseline
           pred_levels = states_levels[:, -1]

           return pred_surface, pred_levels

.. seealso::

   :py:class:`~ocean_flow.modules.train_module.TrainingModule`
       Abstract base class for training workflows.

   :py:func:`~ocean_flow.modules.utils.process_inputs`
       Utility for flattening time × variable input windows.

   :doc:`/guides/forecasting`
       User guide covering the full forecast pipeline.
