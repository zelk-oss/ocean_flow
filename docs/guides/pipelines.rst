Pipelines: Pre/Post Transform Composition
=========================================

The :py:mod:`ocean_flow.pipelines` package defines composable preprocessing and
postprocessing building blocks used by both training and forecasting modules.
Pipelines convert between physical-space variables and latent model space while
keeping these transforms modular and testable.

Purpose and responsibilities
----------------------------

Pipeline components are responsible for:

- transforming inputs before they are passed to neural networks,
- transforming model outputs back to physical-space predictions,
- transforming physical targets to latent space for training losses, and
- enforcing variable-level constraints (for example lower bounds).

Base modules and composition model
----------------------------------

The package defines two abstract base interfaces:

- :py:class:`~ocean_flow.pipelines.pre_module.PreModule` with a ``forward`` method,
- :py:class:`~ocean_flow.pipelines.post_module.PostModule` with ``forward`` and
  ``to_latent`` methods.

Composition is provided by:

- :py:class:`~ocean_flow.pipelines.pipelines.PrePipeline`: applies submodules in
  insertion order,
- :py:class:`~ocean_flow.pipelines.pipelines.PostPipeline`: applies ``forward`` in
  insertion order and ``to_latent`` in reverse order.

This ordering makes postprocessing chains usable in both directions during
training (target-to-latent) and inference (latent-to-physical).

What is implemented
-------------------

Concrete modules in this repository are:

- :py:class:`~ocean_flow.pipelines.normalization.PreNormalization`:
  channel-wise affine normalization with registered mean/std buffers.
- :py:class:`~ocean_flow.pipelines.tendency.TendencyPrediction`:
  latent tendency <-> physical increment conversion relative to the initial
  state.
- :py:class:`~ocean_flow.pipelines.bounding.LowerBoundPrediction`:
  element-wise lower-bound enforcement in physical space.
- :py:func:`~ocean_flow.pipelines.utils.add_dimensions`:
  helper to add singleton dimensions for broadcastable parameters.

Role in training and forecasting modules
----------------------------------------

- In training, :py:func:`ocean_flow.modules.utils.preprocess_data` applies pre-
  pipelines to input windows and uses post-pipeline ``to_latent`` transforms
  for target preparation.
- In forecasting, forecast modules hold pre/post pipelines as part of module
  state and apply them around network forward steps.
- This separation keeps network classes focused on latent dynamics while
  pipelines own data-domain conversions and physical constraints.

Included instantiation patterns
-------------------------------

Default configs under ``configs/pipelines/`` show the standard pattern:

- ``pre_pipeline.yaml`` instantiates one
  :py:class:`~ocean_flow.pipelines.pipelines.PrePipeline` with variable-family
  branches (``states_surface`` and ``states_levels``), each using
  :py:class:`~ocean_flow.pipelines.normalization.PreNormalization`.
- ``post_pipeline.yaml`` instantiates one top-level
  :py:class:`~ocean_flow.pipelines.pipelines.PostPipeline` with per-variable branches.
  Each branch is itself a :py:class:`~ocean_flow.pipelines.pipelines.PostPipeline`
  chaining:

  1. :py:class:`~ocean_flow.pipelines.tendency.TendencyPrediction`
  2. :py:class:`~ocean_flow.pipelines.bounding.LowerBoundPrediction`

This nested structure enables variable-specific postprocessing while preserving
uniform module interfaces.

Design decisions
----------------

- **ModuleDict-based composition** keeps ordering explicit and serializable via
  Hydra instantiation.
- **Registered buffers for statistics/bounds** ensure tensors move with device
  placement and checkpoints.
- **Bidirectional postprocessing API** (`to_latent` + `forward`) aligns loss
  computation and inference semantics.

Implicit assumptions and constraints
------------------------------------

- input tensors follow expected variable ordering and shape conventions,
- configured ``add_dims`` are valid for broadcasting statistics and bounds,
- postprocessing chains are approximately invertible for training targets where
  required,
- module keys (for example ``states_surface`` and ``states_levels``) match the
  keys used by module utilities and dataloaders.

Extension and implementation instructions
-----------------------------------------

To add a new pre-module:

1. Subclass :py:class:`~ocean_flow.pipelines.pre_module.PreModule`.
2. Implement ``forward`` with shape-preserving or well-documented behavior.
3. Add it to an appropriate :py:class:`~ocean_flow.pipelines.pipelines.PrePipeline`
   branch in config.

To add a new post-module:

1. Subclass :py:class:`~ocean_flow.pipelines.post_module.PostModule`.
2. Implement both ``to_latent`` and ``forward`` consistently.
3. Place it in the intended order within a
   :py:class:`~ocean_flow.pipelines.pipelines.PostPipeline` chain.

After extension:

- update Hydra pipeline configs,
- verify integration through module-level tests,
- add unit tests for both forward and latent-direction behavior.

Minimal custom module examples
------------------------------

Custom pre-module
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from ocean_flow.pipelines.pre_module import PreModule

   class MyPreShift(PreModule):
     def __init__(self, shift: float) -> None:
       super().__init__()
       self.shift = shift

     def forward(
       self,
       in_tensor: torch.Tensor,
       *args,
       **kwargs,
     ) -> torch.Tensor:
       return in_tensor + self.shift

Custom post-module
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from ocean_flow.pipelines.post_module import PostModule

   class MyPostScale(PostModule):
     def __init__(self, scale: float) -> None:
       super().__init__()
       self.scale = scale

     def to_latent(
       self,
       target: torch.Tensor,
       initial: torch.Tensor,
       *args,
       **kwargs,
     ) -> torch.Tensor:
       return target / self.scale

     def forward(
       self,
       prediction: torch.Tensor,
       initial: torch.Tensor,
       *args,
       **kwargs,
     ) -> torch.Tensor:
       return prediction * self.scale

Hydra wiring pattern
~~~~~~~~~~~~~~~~~~~~

In config files under ``configs/pipelines/``, attach custom modules inside
``PrePipeline`` and ``PostPipeline`` branches keyed by variable family names
used by your modules (for example ``states_surface`` and ``states_levels``).
