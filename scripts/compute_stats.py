# scripts/compute_stats.py
import numpy as np
import xarray as xr

ZARR_PATH = "/lustre/fsn1/projects/rech/wbg/ukv59en/ocean_flow_data/data/train.zarr"
VARIABLE = "q"

dt = 7
block_time = 32   # reduce to 16 if still killed, increase to 64 if fine

ds = xr.open_zarr(ZARR_PATH)
q = ds[VARIABLE]

print(q)
print("dims:", q.dims)
print("shape:", q.shape)

n_run = q.sizes["run"]
n_time = q.sizes["time"]

# Running sums for state
state_sum = 0.0
state_sumsq = 0.0
state_count = 0

# Running sums for residual
res_sum = 0.0
res_sumsq = 0.0
res_count = 0

for r in range(n_run):
    print(f"Processing run {r + 1}/{n_run}")

    # ------------------------------------------------------------
    # State stats: q
    # ------------------------------------------------------------
    for t0 in range(0, n_time, block_time):
        t1 = min(t0 + block_time, n_time)

        block = q.isel(run=r, time=slice(t0, t1)).values
        block = block.astype(np.float64, copy=False)

        state_sum += block.sum()
        state_sumsq += np.square(block).sum()
        state_count += block.size

        del block

    # ------------------------------------------------------------
    # Residual stats: q[t + dt] - q[t]
    # computed only inside this run
    # ------------------------------------------------------------
    n_pairs = n_time - dt

    for t0 in range(0, n_pairs, block_time):
        t1 = min(t0 + block_time, n_pairs)

        x0 = q.isel(run=r, time=slice(t0, t1)).values
        x1 = q.isel(run=r, time=slice(t0 + dt, t1 + dt)).values

        res = x1.astype(np.float64, copy=False) - x0.astype(np.float64, copy=False)

        res_sum += res.sum()
        res_sumsq += np.square(res).sum()
        res_count += res.size

        del x0, x1, res

# Final statistics
state_mean = state_sum / state_count
state_var = state_sumsq / state_count - state_mean**2
state_std = np.sqrt(max(state_var, 0.0))

res_mean = res_sum / res_count
res_var = res_sumsq / res_count - res_mean**2
res_std = np.sqrt(max(res_var, 0.0))

print(f"state: mean={state_mean:.6e}, std={state_std:.6e}")
print(f"resid: mean={res_mean:.6e},  std={res_std:.6e}")