# scripts/prepare_data.py
import numpy as np
import xarray as xr
from pathlib import Path

ZARR_PATHS = [
    "/lustre/fsn1/projects/rech/wbg/ukv59en/simu_pyqg_512_3_run0_second_moreflush.zarr",
    "/lustre/fsn1/projects/rech/wbg/ukv59en/simu_pyqg_512_3_run1_second_moreflush.zarr",
    "/lustre/fsn1/projects/rech/wbg/ukv59en/simu_pyqg_512_3_run2_second_moreflush.zarr",
]

VARIABLE = "q"

OUT_TRAIN = "/lustre/fsn1/projects/rech/wbg/ukv59en/ocean_flow_data/data/train.zarr"
OUT_VAL   = "/lustre/fsn1/projects/rech/wbg/ukv59en/ocean_flow_data/data/val.zarr"

VAL_FRAC = 0.1


def open_one_run(path: str, ensemble_id: int) -> xr.Dataset:
    ds = xr.open_zarr(path, chunks="auto")[[VARIABLE]]

    # Clean per-run time coordinate.
    n_time = ds.sizes["time"]
    ds = ds.assign_coords(time=np.arange(n_time))

    # Treat each simulation run as one ensemble member.
    ds = ds.expand_dims(ensemble=[ensemble_id])

    return ds


runs = [
    open_one_run(path, ensemble_id=i)
    for i, path in enumerate(ZARR_PATHS)
]

# Result initially has q(ensemble, time, ...)
ds = xr.concat(
    runs,
    dim="ensemble",
    coords="minimal",
    compat="override",
    combine_attrs="override",
)

# Reorder to template expectation: q(time, ensemble, ...)
other_dims = [d for d in ds[VARIABLE].dims if d not in ("time", "ensemble")]
ds = ds.transpose("time", "ensemble", *other_dims)

print(ds)
print("q dims:", ds[VARIABLE].dims)
print("q shape:", ds[VARIABLE].shape)

# Split along time, preserving all ensemble members.
n_time = ds.sizes["time"]
n_val = int(n_time * VAL_FRAC)

if n_val <= 0:
    raise ValueError(f"n_val={n_val}. Increase VAL_FRAC or use longer simulations.")

train = ds.isel(time=slice(None, n_time - n_val))
val   = ds.isel(time=slice(n_time - n_val, None))

# Chunk with time first, ensemble separate.
train = train.chunk({"time": min(100, train.sizes["time"]), "ensemble": 1})
val   = val.chunk({"time": min(100, val.sizes["time"]), "ensemble": 1})

Path(OUT_TRAIN).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_VAL).parent.mkdir(parents=True, exist_ok=True)

train.to_zarr(OUT_TRAIN, mode="w", zarr_version=2)
val.to_zarr(OUT_VAL, mode="w", zarr_version=2)

print("Done.")
print(f"Wrote train to {OUT_TRAIN}")
print(f"Wrote val   to {OUT_VAL}")