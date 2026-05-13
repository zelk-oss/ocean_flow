# scripts/prepare_test_data.py
import numpy as np
import xarray as xr
from pathlib import Path

TEST_ZARR_PATH = "/lustre/fsn1/projects/rech/wbg/ukv59en/simu_pyqg_512_3_run3_second_moreflush.zarr"

VARIABLE = "q"

OUT_TEST = "/lustre/fsn1/projects/rech/wbg/ukv59en/ocean_flow_data/data/test.zarr"


def open_test_run(path: str, ensemble_id: int = 0) -> xr.Dataset:
    ds = xr.open_zarr(path, chunks="auto")[[VARIABLE]]

    # Clean per-run time coordinate.
    n_time = ds.sizes["time"]
    ds = ds.assign_coords(time=np.arange(n_time))

    # Treat this simulation run as one ensemble member.
    ds = ds.expand_dims(ensemble=[ensemble_id])

    return ds


def reorder_to_template(ds: xr.Dataset) -> xr.Dataset:
    """
    Reorder dataset to template expectation:
        q(time, ensemble, ...)
    """
    other_dims = [d for d in ds[VARIABLE].dims if d not in ("time", "ensemble")]
    return ds.transpose("time", "ensemble", *other_dims)


def main():
    test = open_test_run(TEST_ZARR_PATH, ensemble_id=0)
    test = reorder_to_template(test)

    print("Test dataset:")
    print(test)
    print("q dims:", test[VARIABLE].dims)
    print("q shape:", test[VARIABLE].shape)

    # Chunk with time first, ensemble separate.
    test = test.chunk({
        "time": min(100, test.sizes["time"]),
        "ensemble": 1,
    })

    Path(OUT_TEST).parent.mkdir(parents=True, exist_ok=True)

    test.to_zarr(OUT_TEST, mode="w", zarr_version=2)

    print("Done.")
    print(f"Wrote test to {OUT_TEST}")


if __name__ == "__main__":
    main()