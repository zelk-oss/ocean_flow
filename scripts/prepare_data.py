# scripts/prepare_data.py
import xarray as xr
import zarr

ZARR_PATHS = [
    "/lustre/fsn1/projects/rech/wbg/ukv59en/simu_pyqg_512_3_run0_second_moreflush.zarr",
    "/lustre/fsn1/projects/rech/wbg/ukv59en/simu_pyqg_512_3_run1_second_moreflush.zarr",
    "/lustre/fsn1/projects/rech/wbg/ukv59en/simu_pyqg_512_3_run2_second_moreflush.zarr",
]
VARIABLE = "q"          # ← the one variable you want
OUT_TRAIN = "/lustre/fsn1/projects/rech/wbg/ukv59en/ocean_flow_data/data/train.zarr"
OUT_VAL   = "/lustre/fsn1/projects/rech/wbg/ukv59en/ocean_flow_data/data/val.zarr"
VAL_FRAC  = 0.1         # last 10% of time → validation

ds = xr.open_mfdataset(
    ZARR_PATHS,
    engine="zarr",
    combine="nested",
    concat_dim="time"
)[[VARIABLE]]           # drop all other variables here

n = len(ds.time)
n_val = int(n * VAL_FRAC)

ds.isel(time=slice(None, n - n_val)).to_zarr(
    OUT_TRAIN,
    mode="w",
    zarr_version=2
)

ds.isel(time=slice(n - n_val, None)).to_zarr(
    OUT_VAL,
    mode="w",
    zarr_version=2
)
print("Done.")