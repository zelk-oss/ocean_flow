Beginner Example: WeatherBench2 Deterministic Surrogate
=======================================================

This guide walks through the complete beginner example bundled with the
{{cookiecutter.project_name}} project. The example trains and runs a deterministic
single-step surrogate model on ERA5 data prepared according to the
`WeatherBench2 <https://weatherbench2.readthedocs.io>`_ 64x32 benchmark
setup. All code lives under ``examples/beginner/``.

The goal of the example is to demonstrate the minimal end-to-end workflow:
downloading atmospheric data, estimating normalisation statistics, wiring a
transformer network into the framework, and producing forecasts via the
standard :py:mod:`{{cookiecutter.project_slug}}.modules` and :py:mod:`{{cookiecutter.project_slug}}.pipelines` abstractions.

Overview
--------

The beginner example targets a 64x32 equiangular grid (64 longitudes,
32 latitudes) with 6 atmospheric state variables. The surrogate performs
a single autoregressive step: given the state at time *t*, predict the
state at time *t+1* (6 hours later). A Hydra configuration assembles the
data, network, pipelines, and training loop from
``examples/beginner/configs/``.

The example demonstrates:

- downloading and chunking ERA5 data to zarr stores,
- computing per-variable mean and standard deviation for z-score
  normalisation,
- building a compact Vision Transformer-style architecture adapted for
  regular latitude-longitude grids,
- training with cosine-latitude area-weighted MSE loss, and
- running autoregressive forecasts with the shared ``scripts/forecast.py``
  entry point.

Data Preparation
----------------

The ``notebooks/`` directory contains two Jupyter notebooks that cover
data acquisition and normalisation. Run them in order before launching
any training or forecast job.

Downloading ERA5 data
~~~~~~~~~~~~~~~~~~~~~

``notebooks/data_download_wb2.ipynb`` fetches six-hourly ERA5 reanalysis
fields from the WeatherBench2 public Google Cloud Storage bucket and
writes three consolidated zarr stores.

The six state variables selected for this example are:

- **U250** -- zonal wind at 250 hPa
- **V250** -- meridional wind at 250 hPa
- **Z500** -- geopotential at 500 hPa
- **SH700** -- specific humidity at 700 hPa
- **T850** -- temperature at 850 hPa
- **T2m** -- 2-metre air temperature

All variables are on the 64x32 equiangular grid (32 latitudes, 64
longitudes) and are stored in a single zarr array named ``states`` with
the layout ``(time, variable, latitude, longitude)``. Time chunking is
set to ``time=100`` to balance read throughput and memory usage.

The dataset is split by year:

+----------------+---------------------------+
| Split          | Years                     |
+================+===========================+
| Training       | 1979 -- 2017              |
+----------------+---------------------------+
| Validation     | 2018 -- 2019              |
+----------------+---------------------------+
| Test           | 2020 -- 2022              |
+----------------+---------------------------+

Each split is saved as a separate zarr store (``train.zarr``,
``val.zarr``, ``test.zarr``) under a common base directory. The notebook
assumes access to the WeatherBench2 public GCS bucket; a Dask
``LocalCluster`` is used to parallelise the download and rechunking.

Estimating normalisation statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``notebooks/data_normalisation_wb2.ipynb`` opens the training zarr store
and computes two sets of statistics that are later injected into the
pipelines:

1. **Per-variable mean and standard deviation** (``states_mean``,
   ``states_std``) -- computed over all time steps and spatial points,
   used by :py:class:`~{{cookiecutter.project_slug}}.pipelines.normalization.PreNormalization` for
   z-score normalisation of network inputs.

2. **Per-variable standard deviation of 6-hour tendencies**
   (``diff_std``) -- the standard deviation of ``states[t] - states[t-1]``
   across the training set, used by
   :py:class:`~{{cookiecutter.project_slug}}.pipelines.tendency.TendencyPrediction` to scale the
   network output back to physical units.

The notebook saves these arrays to a file that is passed to the Hydra
configs via the ``pipeline@pre_pipeline`` and
``pipeline@post_pipeline`` config groups.

.. note::

   Always estimate normalisation statistics on the **training split
   only**. Using validation or test data would leak future information
   into the normalization and invalidate evaluation metrics.

Network Architecture
--------------------

The network is implemented in
``examples/beginner/networks/transformer.py`` as the ``WB2Transformer``
class. It is a compact Vision Transformer adapted for regular
latitude-longitude grids.

Input and output shapes are both ``(B, C, 32, 64)`` where ``B`` is the
batch size and ``C`` is the number of atmospheric channels. The default
configuration sets ``C = 7`` (``n_input = n_output = 7``).

The forward pass proceeds in four stages.

Patch tokenizer
~~~~~~~~~~~~~~~

A single ``nn.Conv2d`` layer with kernel size 2 and stride 2 projects
the input field from ``(B, C_in, 32, 64)`` to a token grid of shape
``(B, F, 16, 32)`` where ``F`` is the model dimension (``n_features``,
default 512). Stride-2 patching halves both spatial dimensions, producing
512 tokens per sample.

Optionally, the tokenizer can prepend a set of learnable tokens to the
sequence; these are projected and added to every token embedding. They
can learn to represent static geographic context (e.g. land/sea mask or
orography) that is shared across all positions.

.. code-block:: python

   # stride-2 patch embedding
   self.tokenizer = nn.Conv2d(
       n_input, n_features, kernel_size=2, stride=2,
   )

Geographic Position Encoding (Spherical RoPE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of adding absolute positional features to the token embeddings,
the model uses **spherical Rotary Position Embeddings (RoPE)** to encode
geographic location. RoPE applies learned rotations to the query and key
vectors in the attention mechanism, with rotation angles derived from
spherical harmonics evaluated at each token's geographic position.

For each of the 512 token positions on the 16x32 token grid, real-valued
spherical harmonics :math:`Y_{lm}` are computed for all degrees
:math:`l = 0, \ldots, \text{max\_l}` and orders :math:`m = -l, \ldots, l`.
With the default ``max_l=7``, this produces 64 basis functions that
capture geographic structure at multiple spatial scales.

The rotation angles for each token position :math:`t`, attention head
:math:`h`, and feature pair :math:`f` are computed as:

.. math::

   \theta_{t,h,f} = \alpha \sum_{m} Y_m(t) \cdot W_{h,m,f}

where :math:`Y_m(t)` are the spherical harmonics at position :math:`t`,
:math:`W` are learnable weights, and :math:`\alpha` is a scaling factor
(default 1.0).

The rotation is applied by splitting each head's features into even/odd
pairs and applying a 2D rotation:

.. math::

   \begin{pmatrix} x'_{2k} \\ x'_{2k+1} \end{pmatrix}
   =
   \begin{pmatrix} \cos\theta_k & -\sin\theta_k \\
                   \sin\theta_k &  \cos\theta_k \end{pmatrix}
   \begin{pmatrix} x_{2k} \\ x_{2k+1} \end{pmatrix}

This approach has several advantages over absolute positional encoding:

- Naturally respects the spherical geometry of the latitude-longitude grid
- Provides position-aware attention without explicit position embeddings
  in the token stream
- Generalises across different grid resolutions through the spherical
  harmonics basis

The colatitude and longitude values used for the 16x32 token grid are
pre-computed and stored as constants (``_COLATS`` and ``_LONS``).

Transformer blocks
~~~~~~~~~~~~~~~~~~

The backbone consists of ``n_blocks`` identical transformer blocks
(default 8). Each block applies the following operations with residual
connections:

1. **Pre-norm self-attention with spherical RoPE** -- RMSNorm is applied
   to the token sequence before computing multi-head self-attention.
   QK-norm (a separate RMSNorm on each head's query and key vectors)
   stabilises training. Spherical RoPE is then applied to the query and
   key vectors to encode geographic position.
2. **Gated MLP** -- the token sequence passes through a pre-norm gated
   SiLU block. The input projection expands the dimension to
   ``2 * mult * n_features``; the output is the element-wise product
   ``SiLU(gate) * value``, projected back to ``n_features``.

Both the attention output projection and the MLP output projection are
initialised with a very small weight standard deviation (``1e-6``) so
that each block starts as a near-identity residual.

.. code-block:: python

   # forward pass of a single transformer block
   x = x + self._attn(self.norm1(x))   # self-attention with RoPE
   x = x + self._mlp(self.norm2(x))    # gated MLP

When the optional ``flash_attn`` package is installed, FlashAttention is
used automatically in place of ``torch.nn.functional.scaled_dot_product_attention``.

Pixel-shuffle head
~~~~~~~~~~~~~~~~~~

After the final transformer block, a linear head projects each token from
``F`` to ``n_output * 4`` channels. Pixel shuffle (einops rearrange) then
reassembles the 512 tokens back to the full spatial resolution:

.. math::

   (B,\;16 \times 32,\;C_{out} \times 4)
   \;\longrightarrow\;
   (B,\;C_{out},\;32,\;64)

The head weights are initialised with a nearest-neighbour pattern: the
four sub-pixels for each output channel share identical initial weights.
This provides a stable starting point before the network learns
spatially varying refinements.

.. code-block:: python

   net = WB2Transformer(
       n_input=7, n_output=7,
       n_features=512, n_blocks=8,
       n_heads=8, mult=2,
   )
   x = torch.randn(2, 7, 32, 64)
   out = net(x)  # torch.Size([2, 7, 32, 64])

Training
--------

Training is configured via Hydra. The relevant config files are:

- ``examples/beginner/configs/module/train_module.yaml`` -- module class
  and optimiser hyperparameters
- ``examples/beginner/configs/network/transformer.yaml`` -- network
  architecture
- ``examples/beginner/configs/pipeline/wb2_pre.yaml`` -- input
  normalisation pipeline
- ``examples/beginner/configs/pipeline/wb2_post.yaml`` -- output
  tendency-prediction pipeline
- ``examples/beginner/configs/data/wb2_64x32.yaml`` -- dataset paths
  and data loader settings

DeterministicTrain
~~~~~~~~~~~~~~~~~~

The training module is implemented in
``examples/beginner/modules/deterministic_train.py`` as the
:py:class:`~examples.beginner.modules.deterministic_train.DeterministicTrain`
class, which extends
:py:class:`~{{cookiecutter.project_slug}}.modules.train_module.TrainingModule`.

:py:class:`~examples.beginner.modules.deterministic_train.DeterministicTrain`
implements the abstract ``estimate_loss`` method with a cosine-latitude
area-weighted mean squared error:

1. The input states ``batch["states"]`` have shape
   ``(B, T+1, C, H, W)``. The last time step is the regression target;
   the preceding steps are network inputs.
2. Inputs are z-score normalised by the ``"states"`` branch of
   :py:class:`~{{cookiecutter.project_slug}}.pipelines.PrePipeline`.
3. The time and channel dimensions are flattened to produce a
   ``(B, T*C, H, W)`` tensor, which is passed to the network.
4. The target is projected to latent space by
   ``post_pipeline["states"].to_latent``.
5. The loss is the weighted spatial mean of the squared error:

   .. math::

      \mathcal{L} = \frac{1}{BCWH}
      \sum_{b,c,h,w} w_h \bigl(\hat{x}_{b,c,h,w} - x_{b,c,h,w}\bigr)^2

   where :math:`w_h = \cos(\phi_h)` is the cosine-latitude weight for
   row *h* (proportional to the area of each grid cell).

Cosine-latitude weighting ensures that grid cells near the equator, which
represent larger physical areas, contribute more to the loss than
high-latitude cells. Without this correction, the dense polar grid would
dominate training and the model would waste capacity on regions of less
physical relevance to global skill scores.

Optimiser and scheduler
~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~{{cookiecutter.project_slug}}.modules.train_module.TrainingModule` provides a shared
AdamW + cosine annealing schedule with linear warm-up. The beginner
configuration uses:

- ``lr = 3e-4``
- ``weight_decay = 0.1``
- ``lr_warmup_steps = 5000``
- ``total_steps = 100000``
- ``ema_rate = 0.9999``

Parameters without weight decay (biases, normalization layers) are
separated into a dedicated parameter group via
:py:func:`~{{cookiecutter.project_slug}}.modules.utils.split_wd_params`.

Running training
~~~~~~~~~~~~~~~~

With the beginner configs registered, training is launched with
``scripts/train.py``:

.. code-block:: bash

   python scripts/train.py \
       "module@train_module=train_module" \
       network=transformer \
       data=wb2_64x32 \
       "pipeline@pre_pipeline=wb2_pre" \
       "pipeline@post_pipeline=wb2_post" \
       data.train_path=/path/to/train.zarr \
       data.val_path=/path/to/val.zarr

The ``mean``, ``std``, and ``diff_std`` arrays estimated in the
normalisation notebook must be supplied to the pipeline configs. The
simplest approach is to add ``pre_pipeline.states.mean=...`` and
``pre_pipeline.states.std=...`` overrides on the command line, or to
write concrete config files that hard-code the values computed for your
dataset.

Forecasting
-----------

The forecast module is implemented in
``examples/beginner/modules/deterministic_forecast.py`` as the
:py:class:`~examples.beginner.modules.deterministic_forecast.DeterministicForecast`
class, which extends
:py:class:`~{{cookiecutter.project_slug}}.modules.forecast_module.ForecastModule`.

DeterministicForecast
~~~~~~~~~~~~~~~~~~~~~

:py:class:`~examples.beginner.modules.deterministic_forecast.DeterministicForecast`
requires no additional ``__init__`` logic beyond what
:py:class:`~{{cookiecutter.project_slug}}.modules.forecast_module.ForecastModule` already provides
(network, pre_pipeline, post_pipeline). The ``forward`` method performs
a single deterministic step:

1. **Pre-processing** -- input states of shape ``(B, T, C, H, W)`` are
   normalised via ``pre_pipeline["states"]``.
2. **Network call** -- time and channel dimensions are flattened to
   ``(B, T*C, H, W)`` before being passed to the network.
3. **Post-processing** -- the latent prediction from the network is
   mapped back to physical space by ``post_pipeline["states"]``, using
   the last input frame as the initial condition for the tendency-based
   inverse transform:

   .. math::

      \hat{x}_{t+1} = x_t + \sigma_{\Delta} \cdot f_\theta(z_t)

   where :math:`\sigma_{\Delta}` is the per-variable standard deviation
   of 6-hour tendencies and :math:`f_\theta` is the network output in
   latent space.

.. code-block:: python

   import torch
   from examples.beginner.modules.deterministic_forecast import (
       DeterministicForecast,
   )

   # states: (B, T, C, H, W)
   states = torch.randn(2, 1, 7, 32, 64)

   # module is loaded from a checkpoint via Hydra + scripts/forecast.py
   # or constructed manually for testing:
   module = DeterministicForecast(network, pre_pipeline, post_pipeline)
   next_state = module(states)  # (2, 7, 32, 64)

Using the forecast script
~~~~~~~~~~~~~~~~~~~~~~~~~

For autoregressive multi-step rollouts and zarr output, wire the module
through ``scripts/forecast.py`` using the
``examples/beginner/configs/module/forecast_module.yaml`` config group:

.. code-block:: bash

   python scripts/forecast.py \
       "module@forecast_module=forecast_module" \
       data=wb2_64x32 \
       "pipeline@pre_pipeline=wb2_pre" \
       "pipeline@post_pipeline=wb2_post" \
       ckpt_path=/path/to/checkpoint.ckpt \
       data.test_path=/path/to/test.zarr \
       init_start="2020-01-01" \
       init_end="2020-12-31" \
       init_freq="1D" \
       lead_time="240h" \
       step_freq="6h"

The forecast script handles checkpoint loading, Dask actor construction,
and writing predictions to a zarr output store indexed by
``(init_time, lead_time)``. See :doc:`/guides/forecasting` for a
detailed description of the forecast pipeline.

See also
--------

- :doc:`/guides/modules` for the abstract training and forecast module
  interfaces that ``DeterministicTrain`` and ``DeterministicForecast``
  extend.
- :doc:`/guides/pipelines` for an explanation of the pre/post pipeline
  design and the available normalisation and tendency modules.
- :doc:`/guides/data_loading` for details on the zarr-based data loader
  used during training.
- :doc:`/api/{{cookiecutter.project_slug}}.modules` for the API reference of
  :py:class:`~{{cookiecutter.project_slug}}.modules.train_module.TrainingModule` and
  :py:class:`~{{cookiecutter.project_slug}}.modules.forecast_module.ForecastModule`.
