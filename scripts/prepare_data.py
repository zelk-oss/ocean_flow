#!/bin/env python
# -*- coding: utf-8 -*-
#
# Prepare train, val, and test zarr stores from raw pyqg simulation runs.
#
# Optimized for large source zarrs containing q, u, v.
# Only q is opened/written.
#
# Strategy:
#   - Do NOT concatenate run0/run1/run2 in memory.
#   - Do NOT open/write u and v.
#   - Create output zarr metadata first.
#   - Write one run at a time into the ensemble dimension using region writes.
#
# Output:
#   train.zarr: run0/run1/run2 as ensemble members, first 90% of time
#   val.zarr:   run0/run1/run2 as ensemble members, last 10% of time
#   test.zarr:  run3 as one independent ensemble member
#
# Time:
#   train time starts at 1970-01-01
#   val time continues after train
#   test time starts again at 1970-01-01

import gc
import shutil
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZARR_PATHS_TRAINVAL = [
    "/lustre/fsn1/projects/rech/wbg/ukv59en/simu_pyqg_512_3_run0_second_moreflush.zarr",
    "/lustre/fsn1/projects/rech/wbg/ukv59en/simu_pyqg_512_3_run1_second_moreflush.zarr",
    "/lustre/fsn1/projects/rech/wbg/ukv59en/simu_pyqg_512_3_run2_second_moreflush.zarr",
]

ZARR_PATH_TEST = (
    "/lustre/fsn1/projects/rech/wbg/ukv59en/"
    "simu_pyqg_512_3_run3_second_moreflush.zarr"
)

VARIABLE = "q"

# Variables to ignore completely when opening source zarrs.
DROP_VARIABLES = ["u", "v"]

OUT_DIR   = Path("/lustre/fsn1/projects/rech/wbg/ukv59en/ocean_flow_data1/data")
OUT_TRAIN = OUT_DIR / "train.zarr"
OUT_VAL   = OUT_DIR / "val.zarr"
OUT_TEST  = OUT_DIR / "test.zarr"

TIME_ORIGIN = "1970-01-01"
TIME_FREQ   = "1D"

VAL_FRAC = 0.1

# For 512 x 512 fields, keep this small.
# If the job is still killed, try 5 or even 2.
TIME_CHUNK = 10

# Each run is one ensemble member.
ENSEMBLE_CHUNK = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fake_time(n_time: int, offset_days: int = 0) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(TIME_ORIGIN) + pd.Timedelta(days=offset_days),
        periods=n_time,
        freq=TIME_FREQ,
    )


def clean_output(path: Path) -> None:
    if path.exists():
        print(f"Removing existing output store: {path}")
        shutil.rmtree(path)


def open_source_q(path: str) -> xr.Dataset:
    """
    Open one source run lazily and keep only q.

    u and v are explicitly dropped so xarray does not even build lazy arrays
    for them.
    """
    ds = xr.open_zarr(
        path,
        chunks={"time": TIME_CHUNK},
        drop_variables=DROP_VARIABLES,
    )

    if VARIABLE not in ds:
        raise KeyError(
            f"Variable {VARIABLE!r} not found in {path}. "
            f"Available variables are: {list(ds.data_vars)}"
        )

    return ds[[VARIABLE]]


def inspect_source(path: str):
    """
    Inspect one source zarr to recover dimensions, sizes, dtype and coords.

    This opens only metadata and only q.
    """
    ds = xr.open_zarr(
        path,
        chunks={},
        drop_variables=DROP_VARIABLES,
    )

    if VARIABLE not in ds:
        raise KeyError(
            f"Variable {VARIABLE!r} not found in {path}. "
            f"Available variables are: {list(ds.data_vars)}"
        )

    ds = ds[[VARIABLE]]
    q = ds[VARIABLE]

    if "time" not in q.dims:
        raise ValueError(f"{VARIABLE!r} has no time dimension. Dims: {q.dims}")

    source_dims = q.dims
    other_dims = [d for d in source_dims if d != "time"]
    other_sizes = {d: ds.sizes[d] for d in other_dims}
    dtype = q.dtype
    n_time = ds.sizes["time"]

    other_coords = {}
    for d in other_dims:
        if d in ds.coords:
            other_coords[d] = ds.coords[d].values
        else:
            other_coords[d] = np.arange(ds.sizes[d])

    ds.close()

    return {
        "source_dims": source_dims,
        "other_dims": other_dims,
        "other_sizes": other_sizes,
        "other_coords": other_coords,
        "dtype": dtype,
        "n_time": n_time,
    }


def make_empty_store(
    path: Path,
    n_time: int,
    ensemble_values,
    other_dims,
    other_sizes,
    other_coords,
    dtype,
    time_offset_days: int,
) -> None:
    """
    Create the output zarr store metadata.

    Important:
    We call to_zarr(..., compute=False) and do NOT compute the delayed object.
    This creates the zarr metadata without filling the whole array with data.
    The actual q chunks are written later by region writes.
    """
    clean_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dims = ("time", "ensemble", *other_dims)

    shape = (
        n_time,
        len(ensemble_values),
        *[other_sizes[d] for d in other_dims],
    )

    chunks = (
        min(TIME_CHUNK, n_time),
        min(ENSEMBLE_CHUNK, len(ensemble_values)),
        *[other_sizes[d] for d in other_dims],
    )

    coords = {
        "time": fake_time(n_time, offset_days=time_offset_days),
        "ensemble": np.asarray(ensemble_values),
    }

    for d in other_dims:
        coords[d] = other_coords[d]

    data = da.empty(
        shape,
        chunks=chunks,
        dtype=dtype,
    )

    ds_template = xr.Dataset(
        {
            VARIABLE: (dims, data),
        },
        coords=coords,
    )

    print(f"\nCreating output store metadata: {path}")
    print(ds_template)
    print(f"Target chunks for {VARIABLE}: {chunks}")

    delayed = ds_template.to_zarr(
        str(path),
        mode="w",
        compute=False,
        zarr_version=2,
    )

    # Do NOT call delayed.compute().
    # That would write the entire empty array and can be huge.

    del delayed, ds_template, data
    gc.collect()


def prepare_piece_for_region_write(
    source_path: str,
    source_time_slice: slice,
    output_time_offset: int,
    expected_n_time: int,
    ensemble_value: int,
    other_dims,
) -> xr.Dataset:
    """
    Open one source run lazily, select a time slice, add one ensemble dimension,
    assign fake datetime coordinates, transpose to final layout, and return
    only the q variable without coordinates.

    Returning no coordinates avoids region-write conflicts with already-created
    coordinate variables in the target store.
    """
    ds = open_source_q(source_path)

    ds = ds.isel(time=source_time_slice)

    n_piece = ds.sizes["time"]

    if n_piece != expected_n_time:
        raise ValueError(
            f"Piece from {source_path} has {n_piece} time steps, "
            f"expected {expected_n_time}."
        )

    ds = ds.assign_coords(
        time=fake_time(n_piece, offset_days=output_time_offset)
    )

    ds = ds.expand_dims(ensemble=[ensemble_value])

    ds = ds.transpose("time", "ensemble", *other_dims)

    ds = ds.chunk({
        "time": min(TIME_CHUNK, ds.sizes["time"]),
        "ensemble": 1,
    })

    # Critical: strip coordinates from the object being region-written.
    # The output store already has the correct coordinates.
    q = ds[VARIABLE]
    ds_out = xr.Dataset(
        {
            VARIABLE: (q.dims, q.data),
        }
    )

    ds.close()
    del ds, q
    gc.collect()

    return ds_out


def write_region(
    ds_piece: xr.Dataset,
    out_path: Path,
    time_region: slice,
    ensemble_region: slice,
) -> None:
    """
    Write a q-only piece into an existing output zarr store.
    """
    ds_piece.to_zarr(
        str(out_path),
        mode="r+",
        region={
            "time": time_region,
            "ensemble": ensemble_region,
        },
        safe_chunks=False,
        zarr_version=2,
    )


def check_compatible_sources(reference_info, path: str) -> None:
    """
    Check that another source run has the same q layout as the reference run.
    """
    info = inspect_source(path)

    if info["source_dims"] != reference_info["source_dims"]:
        raise ValueError(
            f"Source dims mismatch for {path}:\n"
            f"  reference: {reference_info['source_dims']}\n"
            f"  current:   {info['source_dims']}"
        )

    if info["other_sizes"] != reference_info["other_sizes"]:
        raise ValueError(
            f"Spatial/non-time sizes mismatch for {path}:\n"
            f"  reference: {reference_info['other_sizes']}\n"
            f"  current:   {info['other_sizes']}"
        )

    if info["n_time"] != reference_info["n_time"]:
        raise ValueError(
            f"Time length mismatch for {path}:\n"
            f"  reference: {reference_info['n_time']}\n"
            f"  current:   {info['n_time']}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Keep execution conservative. Avoid dask launching many chunks in parallel.
    dask.config.set(scheduler="single-threaded")

    print("Inspecting reference source run...")
    ref = inspect_source(ZARR_PATHS_TRAINVAL[0])

    source_dims = ref["source_dims"]
    other_dims = ref["other_dims"]
    other_sizes = ref["other_sizes"]
    other_coords = ref["other_coords"]
    dtype = ref["dtype"]
    n_time = ref["n_time"]

    print("\nReference q metadata:")
    print(f"  source dims: {source_dims}")
    print(f"  other dims:  {other_dims}")
    print(f"  other sizes: {other_sizes}")
    print(f"  dtype:       {dtype}")
    print(f"  n_time:      {n_time}")

    print("\nChecking run0/run1/run2 compatibility...")
    for path in ZARR_PATHS_TRAINVAL[1:]:
        check_compatible_sources(ref, path)

    n_val = max(1, int(n_time * VAL_FRAC))
    n_train = n_time - n_val

    if n_train <= 0:
        raise ValueError(
            f"n_train={n_train} after reserving {n_val} validation steps."
        )

    print("\nTrain/val split:")
    print(f"  train time steps: {n_train}")
    print(f"  val time steps:   {n_val}")
    print(f"  total time steps: {n_time}")

    train_ensembles = list(range(len(ZARR_PATHS_TRAINVAL)))

    # ------------------------------------------------------------------
    # Create output zarr stores
    # ------------------------------------------------------------------

    make_empty_store(
        OUT_TRAIN,
        n_time=n_train,
        ensemble_values=train_ensembles,
        other_dims=other_dims,
        other_sizes=other_sizes,
        other_coords=other_coords,
        dtype=dtype,
        time_offset_days=0,
    )

    make_empty_store(
        OUT_VAL,
        n_time=n_val,
        ensemble_values=train_ensembles,
        other_dims=other_dims,
        other_sizes=other_sizes,
        other_coords=other_coords,
        dtype=dtype,
        time_offset_days=n_train,
    )

    # ------------------------------------------------------------------
    # Fill train and val one run at a time
    # ------------------------------------------------------------------

    for ensemble_id, source_path in enumerate(ZARR_PATHS_TRAINVAL):
        print("\n" + "-" * 72)
        print(f"Processing source run {ensemble_id}")
        print(source_path)

        # Train piece
        print(f"Writing train.zarr, ensemble={ensemble_id}")

        ds_train_piece = prepare_piece_for_region_write(
            source_path=source_path,
            source_time_slice=slice(0, n_train),
            output_time_offset=0,
            expected_n_time=n_train,
            ensemble_value=ensemble_id,
            other_dims=other_dims,
        )

        write_region(
            ds_piece=ds_train_piece,
            out_path=OUT_TRAIN,
            time_region=slice(0, n_train),
            ensemble_region=slice(ensemble_id, ensemble_id + 1),
        )

        del ds_train_piece
        gc.collect()

        # Val piece
        print(f"Writing val.zarr, ensemble={ensemble_id}")

        ds_val_piece = prepare_piece_for_region_write(
            source_path=source_path,
            source_time_slice=slice(n_train, n_time),
            output_time_offset=n_train,
            expected_n_time=n_val,
            ensemble_value=ensemble_id,
            other_dims=other_dims,
        )

        write_region(
            ds_piece=ds_val_piece,
            out_path=OUT_VAL,
            time_region=slice(0, n_val),
            ensemble_region=slice(ensemble_id, ensemble_id + 1),
        )

        del ds_val_piece
        gc.collect()

    # ------------------------------------------------------------------
    # Test store from run3
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("Preparing independent test set from run3")

    test_info = inspect_source(ZARR_PATH_TEST)

    if test_info["other_dims"] != other_dims:
        raise ValueError(
            f"Test other dims differ from train/val:\n"
            f"  train/val: {other_dims}\n"
            f"  test:      {test_info['other_dims']}"
        )

    if test_info["other_sizes"] != other_sizes:
        raise ValueError(
            f"Test spatial/non-time sizes differ from train/val:\n"
            f"  train/val: {other_sizes}\n"
            f"  test:      {test_info['other_sizes']}"
        )

    n_test = test_info["n_time"]

    make_empty_store(
        OUT_TEST,
        n_time=n_test,
        ensemble_values=[3],
        other_dims=other_dims,
        other_sizes=other_sizes,
        other_coords=other_coords,
        dtype=test_info["dtype"],
        time_offset_days=0,
    )

    print("Writing test.zarr, ensemble=3")

    ds_test_piece = prepare_piece_for_region_write(
        source_path=ZARR_PATH_TEST,
        source_time_slice=slice(0, n_test),
        output_time_offset=0,
        expected_n_time=n_test,
        ensemble_value=3,
        other_dims=other_dims,
    )

    write_region(
        ds_piece=ds_test_piece,
        out_path=OUT_TEST,
        time_region=slice(0, n_test),
        ensemble_region=slice(0, 1),
    )

    del ds_test_piece
    gc.collect()

    print("\nDone.")
    print(f"Wrote train: {OUT_TRAIN}")
    print(f"Wrote val:   {OUT_VAL}")
    print(f"Wrote test:  {OUT_TEST}")


if __name__ == "__main__":
    main()