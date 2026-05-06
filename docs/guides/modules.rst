Modules: Training and Forecast Orchestration
============================================

The :py:mod:`ocean_flow.modules` package provides high-level PyTorch Lightning
abstractions that connect neural networks, preprocessing/postprocessing
pipelines, and train/forecast execution logic.

Purpose and responsibilities
----------------------------

The package is responsible for:

- defining reusable training and forecasting module interfaces,
- centralizing shared loop/optimizer behavior,
- keeping data-transform logic delegated to :py:mod:`ocean_flow.pipelines`, and
- providing utilities for tensor preparation and optimizer parameter groups.

What is implemented
-------------------

The current package includes three modules:

- :py:mod:`ocean_flow.modules.forecast_module`:

  - :py:class:`~ocean_flow.modules.forecast_module.ForecastModule` is a pure abstract
    base class for single-step forecast modules.
  - Subclasses implement
    :py:meth:`~ocean_flow.modules.forecast_module.ForecastModule.forward`; all
    prediction-loop and zarr I/O logic lives outside this class in
    ``ForecastInference`` or ``scripts/forecast.py``.

- :py:mod:`ocean_flow.modules.train_module`:

  - :py:class:`~ocean_flow.modules.train_module.TrainingModule` is an abstract base
    for optimization logic and Lightning hooks.
  - it implements common training/validation flow, EMA updates, and optimizer
    setup with AdamW + cosine warm restarts.

- :py:mod:`ocean_flow.modules.utils`:

  - :py:func:`~ocean_flow.modules.utils.process_inputs` and
    :py:func:`~ocean_flow.modules.utils.preprocess_data` convert state windows and
    targets into model-ready latent tensors.
  - :py:func:`~ocean_flow.modules.utils.split_wd_params` builds decayed vs non-decayed
    parameter groups.

Design decisions
----------------

- **Abstract base classes for workflows**: model-specific objectives and
  rollouts are implemented in subclasses, while shared orchestration stays in
  one place.
- **Pipeline-first transforms**: normalization, tendency conversion, and bounds
  are delegated to :py:mod:`ocean_flow.pipelines` modules instead of being embedded in
  network code.
- **Forecast module as plugin contract**: :py:class:`ForecastModule` defines
  only the plugin interface (``__init__`` + ``forward``); inference pipelines
  and zarr I/O are wired externally so the module stays testable and reusable.
- **Deterministic optimizer grouping**: parameter grouping logic is explicit so
  weight-decay behavior is testable and predictable.

Implicit assumptions and constraints
------------------------------------

- batch dictionaries include keys expected by configured pre/post pipelines,
- output tuple order in ``forward`` matches the expected variable order,
- post-pipeline branches (for example ``states_surface`` and
  ``states_levels``) implement both ``forward`` and ``to_latent``, and
- subclasses of :py:class:`ForecastModule` do not implement prediction
  loops or I/O — those are wired externally via ``scripts/forecast.py``.

Extension and implementation instructions
-----------------------------------------

To add a new training workflow:

1. Subclass :py:class:`~ocean_flow.modules.train_module.TrainingModule`.
2. Implement
   :py:meth:`~ocean_flow.modules.train_module.TrainingModule.estimate_loss`.
3. Optionally override
   :py:meth:`~ocean_flow.modules.train_module.TrainingModule.estimate_auxiliary_losses`
   for extra validation diagnostics.

To add a new forecast workflow:

1. Subclass :py:class:`~ocean_flow.modules.forecast_module.ForecastModule`.
2. Implement ``forward`` for a single prediction step.
3. Wire pre/post pipeline usage inside ``forward`` as required.

For both workflows:

- wire transforms through configured pre/post pipelines, and
- add focused tests for forward-pass behavior and error paths.

Minimal extension templates
---------------------------

The following examples show the smallest useful subclasses for extending the
framework.

Custom training module
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from ocean_flow.modules import TrainingModule
   from ocean_flow.modules.utils import preprocess_data

   class MyTrainingModule(TrainingModule):
     def estimate_loss(
       self,
       batch: dict,
       prefix: str = "train",
     ) -> dict[str, torch.Tensor]:
       input_tensor, latent_surface, latent_levels = preprocess_data(
         states_surface=batch["states_surface"],
         states_levels=batch["states_levels"],
         pre_pipeline=self.pre_pipeline,
         post_pipeline=self.post_pipeline,
       )
       output = self.network(input_tensor)
       pred_surface = output[:, :latent_surface.size(-3)]
       pred_levels = output[:, latent_surface.size(-3):].reshape_as(
         latent_levels,
       )
       loss_surface = torch.nn.functional.mse_loss(pred_surface, latent_surface)
       loss_levels = torch.nn.functional.mse_loss(pred_levels, latent_levels)
       loss = loss_surface + loss_levels
       self.log(f"{prefix}/loss", loss)
       return {"loss": loss}

Custom forecast module
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from ocean_flow.modules import ForecastModule
   from ocean_flow.modules.utils import process_inputs

   class MyForecastModule(ForecastModule):
     def forward(
       self,
       states_surface: torch.Tensor,
       states_levels: torch.Tensor,
     ) -> tuple[torch.Tensor, torch.Tensor]:
       in_tensor = process_inputs(
         states_surface=states_surface,
         states_levels=states_levels,
         pre_pipeline=self.pre_pipeline,
       )
       out_tensor = self.network(in_tensor)
       n_surf = states_surface.shape[2]
       out_surface = self.post_pipeline["states_surface"](
         out_tensor[:, :n_surf], states_surface[:, -1],
       )
       out_levels = states_levels[:, -1]
       return out_surface, out_levels

For full pipeline integration (autoregressive rollout and zarr output),
wire the concrete subclass through ``scripts/forecast.py``.
