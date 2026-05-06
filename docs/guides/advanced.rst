Advanced Example: WeatherBench2 Flow Matching
=============================================

This guide covers the advanced WeatherBench2 example bundled with the
Ocean Flow Matching project. The example demonstrates conditional generative
flow matching: a class of generative model that learns a continuous vector
field transforming noise into atmospheric tendencies. All code lives under
``examples/advanced/``.

Overview
--------

Flow matching generalises deterministic surrogate models to a generative
setting. Instead of predicting a single next state, the model learns to
integrate a learned vector field from a noise sample to a target tendency,
conditioned on the current atmospheric state. This produces diverse,
physically plausible ensemble members from a single model.

The advanced example operates on a richer atmospheric dataset than the
beginner example: six variables sampled at 13 pressure levels, plus five
surface variables, all on the WeatherBench2 64x32 equiangular grid (64
longitudes, 32 latitudes) at 6-hour timesteps. The network is a 3D-aware
transformer with separate horizontal and vertical self-attention blocks,
enabling the model to represent both large-scale horizontal structure and
vertical coupling across pressure levels simultaneously.

Prerequisites
-------------

The following packages must be installed before running any script in this
example.

- ``gcsfs`` — enables access to the WeatherBench2 Google Cloud Storage bucket
  used by the download script.
- ``dask[distributed]`` — provides the ``LocalCluster`` used to parallelise
  data downloading and rechunking.
- ``wandb`` — provides experiment tracking used by the default logger
  configuration; run ``wandb login`` once before starting training to
  authenticate with the Weights & Biases service.

Data Preparation
----------------

ERA5 reanalysis fields are fetched from the WeatherBench2 public Google
Cloud Storage bucket and stored as consolidated zarr splits. Two command-line
scripts handle data acquisition and normalisation statistics estimation.

.. warning::

   Access to the WeatherBench2 GCS bucket requires the ``gcsfs`` package.
   Install it with ``pip install gcsfs``. Depending on the project's access
   policy, authentication may also be required: run
   ``gcloud auth application-default login`` for user credentials, or set the
   ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable to a service
   account key file before invoking the download script.

Variables
^^^^^^^^^

The example selects six level variables, each sampled at 13 pressure levels.

+--------+--------------------------+---------------------------+
| Symbol | Description              | ERA5 variable name        |
+========+==========================+===========================+
| Z      | Geopotential             | geopotential              |
+--------+--------------------------+---------------------------+
| T      | Temperature              | temperature               |
+--------+--------------------------+---------------------------+
| Q      | Specific humidity        | specific_humidity         |
+--------+--------------------------+---------------------------+
| U      | Zonal wind component     | u_component_of_wind       |
+--------+--------------------------+---------------------------+
| V      | Meridional wind component| v_component_of_wind       |
+--------+--------------------------+---------------------------+
| W      | Vertical velocity        | vertical_velocity         |
+--------+--------------------------+---------------------------+

The 13 pressure levels used are (in hPa):
50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000.

Five surface variables complete the input state.

+--------+------------------------------+------------------------------------------+
| Symbol | Description                  | ERA5 variable name                       |
+========+==============================+==========================================+
| T2m    | 2-metre air temperature      | 2m_temperature                           |
+--------+------------------------------+------------------------------------------+
| Dew2m  | 2-metre dewpoint temperature | 2m_dewpoint_temperature                  |
+--------+------------------------------+------------------------------------------+
| U10m   | 10-metre zonal wind          | 10m_u_component_of_wind                  |
+--------+------------------------------+------------------------------------------+
| V10m   | 10-metre meridional wind     | 10m_v_component_of_wind                  |
+--------+------------------------------+------------------------------------------+
| MSLP   | Mean sea-level pressure      | mean_sea_level_pressure                  |
+--------+------------------------------+------------------------------------------+

Splits
^^^^^^

The dataset is divided by year into three non-overlapping splits.

+------------+-------------------+
| Split      | Years             |
+============+===================+
| Training   | 1979 -- 2017      |
+------------+-------------------+
| Validation | 2018              |
+------------+-------------------+
| Test       | 2020 -- 2022      |
+------------+-------------------+

.. note::

   Year 2019 is intentionally excluded from all splits. It serves as a
   safety margin between the validation period (2018) and the test period
   (2020--2022), ensuring no temporal overlap or data leakage between the
   two evaluation sets.

Each split is written as a separate zarr store (``train.zarr``,
``val.zarr``, ``test.zarr``) under a common base directory. Time chunking
is set to ``time=100`` to balance I/O throughput and memory usage. The
data arrays are ``states_levels`` with layout
``(time, variable, level, latitude, longitude)`` and ``states_surface``
with layout ``(time, variable, latitude, longitude)``.

Downloading data
^^^^^^^^^^^^^^^^

The ``download_data.py`` script fetches the ERA5 fields from GCS and writes
the three zarr splits. The script uses a Dask ``LocalCluster`` to parallelise
the download and rechunking.

.. code-block:: bash

   python examples/advanced/download_data.py \
       --store_path /path/to/data/advanced \
       --num_workers 4

Estimating normalisation statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``estimate_normalisation.py`` script opens the training zarr store and
prints six tables of per-variable statistics. All spatial reductions apply
cosine-latitude weighting to prevent equatorial over-representation in global
mean and standard deviation estimates.

.. code-block:: bash

   python examples/advanced/estimate_normalisation.py \
       --train_path /path/to/data/advanced/train.zarr

The script prints per-variable mean, standard deviation, and tendency standard
deviation for both surface and level arrays. These values are then passed to
the pipeline configuration as ``mean``, ``std``, and ``std`` (tendency)
overrides before training.

.. note::

   Always estimate normalisation statistics on the training split only.
   Using validation or test data would leak future information into the
   normalisation and invalidate evaluation metrics.

The printed output must be reformatted as comma-separated Hydra list literals
before use as CLI overrides. Each printed label maps to a specific config
parameter as follows.

+-------------------------------------+-------------------------------------------+
| Printed label                       | Hydra override parameter                  |
+=====================================+===========================================+
| ``states_surface_mean``             | ``pre_pipeline.states_surface.mean``      |
+-------------------------------------+-------------------------------------------+
| ``states_surface_std``              | ``pre_pipeline.states_surface.std``       |
+-------------------------------------+-------------------------------------------+
| ``states_surface_tendency_std``     | ``post_pipeline.states_surface.std``      |
+-------------------------------------+-------------------------------------------+
| ``states_levels_mean``              | ``pre_pipeline.states_levels.mean``       |
+-------------------------------------+-------------------------------------------+
| ``states_levels_std``               | ``pre_pipeline.states_levels.std``        |
+-------------------------------------+-------------------------------------------+
| ``states_levels_tendency_std``      | ``post_pipeline.states_levels.std``       |
+-------------------------------------+-------------------------------------------+

.. note::

   ``pre_pipeline`` ``std`` values normalise the raw input state, while
   ``post_pipeline`` ``std`` values scale the predicted tendency (rate of
   change). These are different physical quantities with different units and
   must not be confused.

Values from each row are collected into a bracket-enclosed, comma-separated
list with no spaces. The following illustrates the expected format.

.. code-block:: bash

   "pre_pipeline.states_surface.mean=[1.0,2.0,3.0,4.0,5.0]"
   "pre_pipeline.states_surface.std=[0.5,0.8,1.2,0.9,1.1]"
   "post_pipeline.states_surface.std=[0.3,0.4,0.6,0.5,0.7]"

The same pattern applies to ``states_levels`` overrides, which will contain
one value per level variable (six values per list).

Flow Matching Theory
--------------------

Flow matching defines a family of generative models based on a linear
interpolant between noise and a target. Given an initial atmospheric state
:math:`x_0` and a target tendency :math:`x_1`, a noised sample
:math:`x_t` at pseudo-time :math:`t \in [0, 1)` is constructed as (here
called *pseudo-time* to distinguish it from the physical forecast lead time
— it parameterises a path in data space between the noise sample and the
target state):

.. math::

   x_t = t \cdot x_1 + (1-t) \cdot \varepsilon,\quad
   \varepsilon \sim \mathcal{N}(0, I)

Here :math:`\varepsilon` is independent Gaussian noise and :math:`t` is
sampled uniformly, using a stratified scheme to reduce estimator variance.
The model is trained to match the analytical velocity :math:`x_1 - \varepsilon`
at each pseudo-time :math:`t`.

The training objective is a latitude-weighted mean squared error between the
network velocity prediction and the analytical velocity:

.. math::

   \mathcal{L} = \mathbb{E}_{t,\, x_0,\, x_1,\, \varepsilon}
   \bigl[
     \lVert
       v_\theta(x_t,\, x_0,\, t) - (x_1 - \varepsilon)
     \rVert^2_{\mathbf{w}}
   \bigr]

The subscript :math:`\mathbf{w}` denotes latitude weighting: each spatial
grid point is weighted by the cosine of its latitude, which is proportional
to its grid-cell area. This ensures high-latitude grid cells, which represent
smaller physical areas, do not disproportionately influence the loss.

At forecast time, a noise sample :math:`z_0 \sim \mathcal{N}(0, I)` is
integrated from :math:`t = 0` to :math:`t = 1` using Euler steps. Each step
queries the trained network for a velocity estimate and advances the state
by a fixed step size :math:`\Delta t`. The result after integration is
decoded by the post-pipeline to recover a physical tendency prediction.

Network Architecture
--------------------

The network is implemented in
``examples/advanced/networks/transformer.py`` as the
:py:class:`~examples.advanced.networks.transformer.Transformer` class.
It is a 3D-aware transformer that processes surface and level fields in a
shared token sequence with separate horizontal and vertical attention.

.. note::

   The network config parameters ``n_surf_in: 10`` and ``n_lev_in: 12`` are
   each double the number of physical variables (5 surface, 6 level). At
   every pseudo-time step the network receives two tensors concatenated on
   the channel dimension: the noisy interpolant :math:`x_t` and the
   normalised conditioning state :math:`x_0`. This concatenation doubles the
   input channel count, giving ``n_surf_in = 5 × 2 = 10`` and
   ``n_lev_in = 6 × 2 = 12``. The output parameters ``n_surf_out: 5`` and
   ``n_lev_out: 6`` match the number of physical variables, as the network
   predicts one velocity per variable.

Tokenisers
^^^^^^^^^^

Two lightweight convolutional tokenisers reduce the spatial resolution from
the full 32x64 grid to a 16x32 token grid (512 tokens per level), using
stride-2 patch embeddings.

The 2D tokeniser handles the surface field. It applies a single
``nn.Conv2d`` layer with kernel size 2 and stride 2, mapping the input
``(B, C_surf, 32, 64)`` to tokens of shape ``(B, 1, 512, F)``, where
``F`` is the model dimension (``n_features``).

The 3D tokeniser handles the level fields. It applies a single
``nn.Conv3d`` layer with kernel size ``(1, 2, 2)`` and stride
``(1, 2, 2)``, mapping ``(B, C_lev, L, 32, 64)`` to tokens of shape
``(B, L, 512, F)``. The level dimension is left unchanged so that vertical
structure is preserved.

The surface and level tokens are concatenated along the level dimension
into a single sequence of shape ``(B, L+1, 512, F)``, with the surface
token appended last at position ``tokens[:, -1]``. This ordering is
invariant to the number of pressure levels.

Attention blocks
^^^^^^^^^^^^^^^^

Each transformer block applies three operations in sequence: horizontal
self-attention, vertical self-attention, and a gated MLP. All attention
sub-modules are conditioned on the pseudo-time embedding produced by
:py:class:`~examples.advanced.networks.transformer.RandomFourierTimeEmbedding`.

:py:class:`~examples.advanced.networks.transformer.HorizontalAttentionBlock`
attends across the 512 spatial token positions at each level independently.
It flattens the batch and level dimensions before computing multi-head
self-attention, then restores them. Position information is encoded with
spherical RoPE using real spherical harmonics as basis functions, matching
the approach used in the beginner transformer. The spherical harmonics are
evaluated at grid positions defined by the module-level constants ``_COLATS``
and ``_LONS`` in ``examples/advanced/networks/transformer.py``, which are
hardcoded for the WeatherBench2 32×64 grid. Users adapting the transformer
to a different resolution must update ``_COLATS`` and ``_LONS`` to match the
new grid's colatitude and longitude coordinates.

:py:class:`~examples.advanced.networks.transformer.VerticalAttentionBlock`
attends across the ``L+1`` level positions at each spatial token
independently. It permutes the tensor to bring the level dimension forward,
flattens the batch and spatial dimensions, computes multi-head
self-attention, and restores the original layout. Position information is
encoded with a simple 1-D
:py:class:`~examples.advanced.networks.transformer.RoPELayer` parameterised
by the log-pressure coordinate of each level.

Both attention blocks use
:py:class:`~examples.advanced.networks.transformer.AdaptiveRMSNorm` (Adaptive
Pre-RMS Norm) as the pre-normalisation layer. Unlike plain RMSNorm, it
projects the pseudo-time embedding to a scale and shift that modulate the
normalised activations, enabling time-dependent conditioning throughout the
network. The projection is zero-initialised so that each block starts as an
unconditional residual near the identity. QK-norm (a separate RMSNorm on
each head's query and key vectors) further stabilises training.

Gated MLP
^^^^^^^^^

Each transformer block ends with a gated MLP. The input is pre-normalised
via :py:class:`~examples.advanced.networks.transformer.AdaptiveRMSNorm`,
then projected to ``2 * n_mult * n_features`` channels. The projection is
split into two halves; the output is the element-wise product
``SiLU(gate) * value``, projected back to ``n_features``. Both the
attention output projection and the MLP output projection are initialised
with small random weights (standard deviation ``1e-4``) so that residual
contributions start near zero.

Pixel-shuffle head
^^^^^^^^^^^^^^^^^^

After the final transformer block, a shared
:py:class:`~examples.advanced.networks.transformer.AdaptiveRMSNorm` is
applied to all ``L+1`` tokens at once. The normalised token sequence is
then split back into level tokens ``tokens[:, :-1]`` and the surface token
``tokens[:, -1]``. Two separate linear projections followed by pixel-shuffle
rearrangement decode each part to the full spatial resolution:

- Surface: ``(B, 512, F)`` :math:`\to` ``(B, n\_surf\_out \times 4, 16, 32)``
  :math:`\to` ``(B, n\_surf\_out, 32, 64)``
- Levels: ``(B, L, 512, F)`` :math:`\to` ``(B, L, n\_lev\_out \times 4, 16, 32)``
  :math:`\to` ``(B, n\_lev\_out, L, 32, 64)``

Using a single shared normalisation pass over all tokens before the split
ensures surface and level outputs receive identical conditioning from the
pseudo-time embedding, with no duplication of normalisation parameters.

Training
--------

Training is configured via Hydra. The relevant config files live under
``examples/advanced/configs/``. The training module is implemented as
:py:class:`~examples.advanced.modules.flow_train.FlowTrainModule`, which
extends :py:class:`~ocean_flow.modules.train_module.TrainingModule` and implements
the flow-matching velocity objective with latitude-weighted loss.

Both ``wb2_pre.yaml`` and ``wb2_post.yaml`` pipeline configs include an
``add_dims`` parameter set to ``[1, 2]``. This instructs the pipeline's
normalisation step to insert size-1 dimensions at positions 1 and 2 of the
statistics tensor, broadcasting the per-variable statistics (shape ``[C]``)
over the time and spatial batch dimensions without explicit tiling.

The ``lat_weights`` argument to ``FlowTrainModule`` must be supplied as a
list of cosine-latitude values for the 32 latitudes. These values are the
cosines of the latitude angles in radians, computed from the training dataset.
The following snippet derives the correct list from the training zarr store.

.. code-block:: python

    import numpy as np, zarr
    ds = zarr.open("/path/to/data/advanced/train.zarr", "r")
    lat = ds["latitude"][:]              # degrees
    lat_weights = np.cos(np.deg2rad(lat))
    print(",".join(f"{w:.6f}" for w in lat_weights))

The printed comma-separated string is passed directly to
``train_module.lat_weights=[v1,v2,...]`` as a Hydra CLI override.

.. note::

   The advanced example training config inherits several config groups
   (``trainer``, ``callbacks``, ``logger``) from the repository-level
   ``configs/`` directory. The training script must be run from the
   repository root so that Hydra can locate both config directories.

.. code-block:: bash

   python scripts/train.py \
       --config-dir examples/advanced/configs \
       data.train_path=/path/to/data/advanced/train.zarr \
       data.val_path=/path/to/data/advanced/val.zarr \
       "train_module.lat_weights=[<values from snippet above>]" \
       "pre_pipeline.states_surface.mean=[...]" \
       "pre_pipeline.states_surface.std=[...]" \
       "pre_pipeline.states_levels.mean=[...]" \
       "pre_pipeline.states_levels.std=[...]" \
       "post_pipeline.states_surface.std=[...]" \
       "post_pipeline.states_levels.std=[...]"

The ``lat_weights`` list must be computed from the training data using the
snippet shown above; the values vary with the grid's latitude coordinates.
The ``mean``, ``std``, and tendency ``std`` values are taken from the output
of ``estimate_normalisation.py``. The configuration sets ``lr=3e-4``,
``weight_decay=0.1``, ``lr_warmup_steps=5000``, ``total_steps=200000``,
and ``ema_rate=0.9999``.

Forecasting
-----------

The forecast module is implemented as
:py:class:`~examples.advanced.modules.flow_forecast.FlowForecastModule`,
which extends :py:class:`~ocean_flow.modules.forecast_module.ForecastModule`. The
module integrates a noise sample from :math:`t=0` to :math:`t=1` using
``n_steps`` Euler steps (default 20), conditioned on the most recent input
state.

The integration schedule is stored as a plain Python attribute
``module.schedule`` (not a registered buffer), so it can be freely
reassigned after construction to change the number of integration steps
without reinstantiating the module. The number of Euler steps is derived
from ``len(schedule) - 1``, so reassigning
``module.schedule = torch.linspace(0, 1, 41)`` automatically changes the
number of steps to 40. Note that ``schedule`` is a plain Python attribute
and is not saved to or loaded from model checkpoints; any reassignment
applies only to the current Python session.

.. code-block:: bash

   python scripts/forecast.py \
       --config-dir examples/advanced/configs \
       ckpt_path=/path/to/data/advanced/best.ckpt \
       data.test_path=/path/to/data/advanced/test.zarr \
       init_start="2020-01-01" \
       init_end="2020-12-31" \
       init_freq="1D" \
       lead_time="240h" \
       step_freq="6h"

The forecast script writes predictions to a zarr store indexed by
``(init_time, lead_time)``. See :doc:`/guides/forecasting` for a full
description of the forecast pipeline and output format.

See Also
--------

- :doc:`/guides/beginner` for the simpler deterministic surrogate example
  that uses the same WeatherBench2 grid and the same framework abstractions.
- :doc:`/guides/modules` for the abstract training and forecast module
  interfaces that :py:class:`~examples.advanced.modules.flow_train.FlowTrainModule`
  and
  :py:class:`~examples.advanced.modules.flow_forecast.FlowForecastModule`
  extend.
- :doc:`/guides/pipelines` for an explanation of the pre/post pipeline
  design and the normalisation and tendency modules used in both examples.
