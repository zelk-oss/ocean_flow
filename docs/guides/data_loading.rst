Data Loading for Training and Validation
=========================================

This guide explains how the Ocean Flow Matching project loads data for
training and validation. The codebase uses a combination of a lightweight
`TrainDataset` class built on top of ``torch.utils.data.Dataset`` and a
``TrainDataModule`` that integrates with PyTorch Lightning. Data are stored in
`zarr` format with one dataset for training and another for validation.

Overview
--------

Training data must be prepared in advance and written to two zarr groups
named ``train.zarr`` and ``val.zarr`` under a common base directory. Each
zarr group contains the following variables:

- ``time`` – the temporal axis (CF‑encoded values).
- 1 or more *state variables* (e.g. temperature, humidity) that the model
  should predict.
- Optional *forcing variables* (e.g. external fluxes) that are treated as
  inputs but not targets.

In addition, static auxiliary fields such as the mesh geometry or land–sea
mask can be supplied from a separate NetCDF file.

The dataset classes handle slicing the time axis into overlapping samples of
fixed length (``n_steps``) with a configurable ``n_step_size`` between
temporal steps. This abstraction keeps the training loop agnostic to the
underlying storage format and allows efficient multi‑worker loading.

.. note::

   Zarr stores must be **consolidated** before use. When creating a store
   from scratch, call ``zarr.consolidate_metadata(store)`` (or use
   ``xarray.Dataset.to_zarr(..., consolidated=True)``) to write the
   ``.zmetadata`` index. Without this, ``TrainDataset`` will raise a
   ``GroupNotFoundError`` on construction.

Data structure
--------------

Instances of ``TrainDataset`` encapsulate a single zarr group. They expose the
following properties (documented in the API reference):

- ``variables`` – list of state+forcing variable names included in each sample.
- ``n_times`` – length of the time dimension present in the dataset.
- ``__len__`` returns the number of possible samples: ``n_times -
  (n_steps - 1) * n_step_size`` per ensemble member. When the zarr group
  contains an ``ensemble`` coordinate (shape ``(n_ensemble,)``), the dataset
  length is ``(n_times - (n_steps - 1) * n_step_size) × n_ensemble``. This
  multiplies the number of available training samples without duplicating data.
- ``__getitem__`` returns a dictionary mapping each variable to a NumPy array
  shaped ``(n_steps, ...)`` and includes a ``"time"`` entry with the raw
  CF‑encoded value at the final input step. The auxiliary variables (if any)
  are also added to the dictionary; they are constant for every index unless
  the auxiliary variable carries an ``ensemble`` first dimension, in which
  case the per-member slice is returned instead.
  When an ensemble dimension is present each flat index ``idx`` is decomposed
  into a time index (``idx // n_ensemble``) and an ensemble-member index
  (``idx % n_ensemble``), so consecutive indices cycle through ensemble
  members before advancing to the next time step.

**Understanding n_steps and n_step_size**

The two parameters jointly define which time indices are selected for each
sample. With a 6-hourly dataset:

- ``n_steps=2, n_step_size=1`` — 2 consecutive steps 6 h apart (12 h window)
- ``n_steps=4, n_step_size=1`` — 4 steps spanning 24 h
- ``n_steps=4, n_step_size=4`` — 4 steps each 24 h apart (72 h window)
- ``n_steps=1, n_step_size=1`` — single-step samples (no multi-step rollout)

The time indices within a sample are
``[t, t + n_step_size, ..., t + (n_steps-1) * n_step_size]``.

TrainDataModule
----------------

``TrainDataModule`` is a subclass of ``lightning.pytorch.LightningDataModule``
and is the recommended entry point for training scripts and unit tests. It
accepts the same parameters that are later forwarded to the underlying
``TrainDataset`` instances, plus several loader‑specific options:

- ``batch_size`` / ``val_batch_size`` – batch sizes for training and
  validation respectively.
- ``n_workers`` – number of worker processes used by the
  :class:`~torch.utils.data.DataLoader`.
- ``pin_memory`` – when ``True`` (the default) memory is pinned on CUDA‑capable
  machines to speed GPU transfers; the flag is ignored when CUDA is unavailable.

The :meth:`~ocean_flow.data.data_module.TrainDataModule.setup` method creates separate
``TrainDataset`` objects for ``train.zarr`` and ``val.zarr`` depending on the
requested stage (``"fit"`` or ``"validate"``). Calling
:func:`~pytorch_lightning.LightningDataModule.prepare_data` before
``setup`` is optional and can be used to download or verify that the datasets
exist.

The :meth:`~ocean_flow.data.data_module.TrainDataModule.train_dataloader` and
:func:`~ocean_flow.data.data_module.TrainDataModule.val_dataloader` methods return
fully configured :class:`~torch.utils.data.DataLoader` objects ready for use in
a training loop. The training loader shuffles the samples and drops the last
discarded partial batch; the validation loader does not shuffle.

Auxiliary data
--------------

If the data module is given ``auxiliary_path`` together with
``auxiliary_variables``, the static fields are loaded once via ``xarray``
and stored in memory. These arrays are added verbatim to every sample. The
code validates that the requested variables exist in the NetCDF file and emits
warnings when one of the two options is supplied without the other.

Ensemble support
----------------

The ``TrainDataset`` natively supports datasets that contain multiple ensemble
members. When the zarr store contains an ``ensemble`` coordinate the class
automatically:

1. Sets ``n_ensemble`` to the length of that coordinate (otherwise ``1``).
2. Scales the dataset length to ``(n_times - (n_steps - 1) * n_step_size) × n_ensemble``.
3. Decomposes each sample index into a time index and an ensemble-member index.

**Zarr store layout with ensemble**

Variables that include the ensemble dimension must have the shape
``(time, ensemble, *spatial_dims)``:

.. code-block:: none

    states_surface: (time, ensemble, variable, latitude, longitude)
    states_levels:  (time, ensemble, variable, level, latitude, longitude)
    ensemble:       (n_ensemble,)   ← required coordinate
    time:           (n_time,)

Variables without an ensemble axis (e.g. ``time``, latitude/longitude
coordinates) are handled correctly: the ensemble index is ignored for those
arrays.

**Auxiliary data with ensemble**

If the auxiliary NetCDF file also contains per-member fields, the first
dimension of those variables must be named ``ensemble`` and have the same
length as the zarr ensemble size. A size mismatch raises a ``ValueError``
during dataset construction. Variables *without* an ensemble dimension in
the auxiliary file are returned as-is for every sample regardless of the
ensemble member.

.. code-block:: python

    # auxiliary.nc layout when ensemble members are present
    # mask:  (ensemble, latitude, longitude)
    # mesh:  (ensemble, channel, latitude, longitude)
    # label: (latitude, longitude)   ← no ensemble, returned unchanged

**Example**

.. code-block:: python

    from ocean_flow.data import TrainDataModule

    dm = TrainDataModule(
        data_path="/data/experiments/ens_wb2",
        state_variables=["states_surface", "states_levels"],
        n_steps=2,
        batch_size=64,
    )
    dm.setup("fit")
    # If the zarr store has 100 time steps and 10 ensemble members:
    # len(dm._train_dataset) == 99 * 10 == 990

Example usage
-------------

.. code-block:: python

   from ocean_flow.data import TrainDataModule

   dm = TrainDataModule(
       data_path="/data/experiments/wb2_64x32",
       state_variables=["temp", "humidity"],
       forcing_variables=["flux"],
       auxiliary_path="/data/mesh.nc",
       auxiliary_variables=["land_mask"],
       n_steps=4,
       n_step_size=1,
       batch_size=128,
       val_batch_size=32,
       n_workers=8,
   )

   # the Lightning trainer will call these automatically, but they can be
   # invoked manually in tests:
   dm.setup("fit")
   train_loader = dm.train_dataloader()
   for batch in train_loader:
       # batch is a dict mapping variable names → (batch, n_steps, ...) arrays
       # it also contains a "time" key with CF-encoded time scalars
       pass

   dm.setup("validate")
   val_loader = dm.val_dataloader()


Best practices
--------------

- **Keep training and validation zarr groups in a single directory.** This
  makes it easier to pass ``data_path`` to scripts and to version the data
  using the same backing store.
- **Tune ``n_workers`` for your hardware.** A value equal to the number of
  physical cores is a good starting point; excessive workers may increase
  memory pressure.
- **Use ``pin_memory=True`` on NVIDIA GPUs.** The flag is a no‑op on CPU
  training but helps unleash the full bandwidth of the device.
- **Shuffle only during training.** The data module takes care of this
  automatically; avoid manually wrapping the loaders with additional
  shuffling layers.
- **Ensemble members are treated as independent samples.** The loader shuffles
  across both time steps and ensemble members; no special sampler is required.
  Make sure ``batch_size`` is chosen to account for the larger effective
  dataset size.

.. note::

   The dataset classes are deliberately simple and do not implement on‑the‑fly
   augmentation or preprocessing. Any such operations should be wrapped in a
   custom ``torch.utils.data.Dataset`` subclass or applied as transforms in the
   training loop.

See also
--------

- :doc:`/api/ocean_flow.data` for the API reference of :class:`~ocean_flow.data.dataset.TrainDataset`
  and :class:`~ocean_flow.data.data_module.TrainDataModule`.
- :doc:`/api/ocean_flow.pipelines` for details on how normalized tensors are produced from
  raw data.
