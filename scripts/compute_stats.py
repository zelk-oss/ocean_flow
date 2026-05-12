# scripts/compute_stats.py
import xarray as xr
import numpy as np

ds = xr.open_zarr("/lustre/fsn1/projects/rech/wbg/ukv59en/ocean_flow_data/data/train.zarr")  # use ONLY train split for stats
q = ds["q"].values  # shape (T, H, W) or (T, nlayers, H, W)

# Stats for the input state
state_mean = float(q.mean())
state_std  = float(q.std())

# Stats for the residual (increment)
# dt here is your n_step_size — the same gap used in TrainDataset
dt = 7  # adjust to your value
residuals = q[dt:] - q[:-dt]
res_mean  = float(residuals.mean())
res_std   = float(residuals.std())

print(f"state: mean={state_mean:.6e}, std={state_std:.6e}")
print(f"resid: mean={res_mean:.6e},  std={res_std:.6e}")