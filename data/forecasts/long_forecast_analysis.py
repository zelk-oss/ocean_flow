#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
forecast_analysis_long.py

Diagnostics for ocean_flow forecast.zarr against real test.zarr truth.
Designed for long forecasts (~1000 lead steps) where materialising the full
4-D array is not viable.

Key changes vs. the original:
  - Full metrics (RMSE, spread) are computed streaming over (ic, lt) — the
    full ensemble is NEVER held in memory simultaneously.
  - Snapshots and GIF use N_SNAP evenly-spaced lead indices only.
  - Section H: spread/skill ratio vs lead time (ratio = spread / RMSE).
"""

import os
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# 0. USER CONFIG
# ═══════════════════════════════════════════════════════════════════

FORECAST_ZARR = (
    "/lustre/fswork/projects/rech/wbg/ukv59en/flow-esm/"
    "surrogate-template/ocean_flow/data/forecasts/"
    "sqg_flow_matching_serious_long_forecast_ens8/forecast.zarr"
)

REFERENCE_ZARR = (
    "/lustre/fsn1/projects/rech/wbg/ukv59en/"
    "ocean_flow_data/data/test.zarr"
)

VAR = "q"
LEV_IDX = 0

OUT_DIR = "sqg_flow_matching_serious_long_forecast_ens8"
os.makedirs(OUT_DIR, exist_ok=True)

# For visualisation only. Metrics are always computed in physical units.
PLOT_NORMALIZED = True

# Which IC to show in snapshot/GIF panels.
REF_IC = 0

# Number of evenly-spaced lead steps shown in snapshots and GIF.
# Metrics (RMSE, spread, spectra, drift) are computed over ALL lead steps.
N_SNAP = 8

# If exact valid-time matching fails allow nearest-neighbour time matching.
ALLOW_NEAREST_TIME = False


# ═══════════════════════════════════════════════════════════════════
# 1. BASIC HELPERS
# ═══════════════════════════════════════════════════════════════════

def to_2d(arr):
    a = np.squeeze(np.asarray(arr, dtype=np.float32))
    if a.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze, got shape {a.shape}")
    return a


def rmse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def as_timedelta64(x):
    return pd.to_timedelta(x)


def infer_time_dim(da):
    preferred = ["time", "init_time", "date", "valid_time"]
    for name in preferred:
        if name in da.dims:
            return name
    for dim in da.dims:
        if dim in da.coords and np.issubdtype(da[dim].dtype, np.datetime64):
            return dim
    raise ValueError(
        f"Could not infer time dimension for {da.name}. "
        f"Dims={da.dims}; coords={list(da.coords)}."
    )


def infer_level_dim(da):
    for name in ["lev", "level", "z", "depth"]:
        if name in da.dims:
            return name
    return None


def select_reference_at_valid_time(ref_da, valid_time, lev_idx=0):
    time_dim = infer_time_dim(ref_da)
    level_dim = infer_level_dim(ref_da)
    if ALLOW_NEAREST_TIME:
        truth = ref_da.sel({time_dim: valid_time}, method="nearest")
    else:
        truth = ref_da.sel({time_dim: valid_time})
    if level_dim is not None:
        truth = truth.isel({level_dim: lev_idx})
    return to_2d(truth.values)


def forecast_lead_days(lead_values):
    days = []
    for x in lead_values:
        td = pd.to_timedelta(x)
        days.append(td / pd.Timedelta(days=1))
    return np.asarray(days, dtype=float)


def maybe_normalize(x, mean, std):
    if not PLOT_NORMALIZED:
        return x
    return (x - mean) / (std + 1e-12)


# ═══════════════════════════════════════════════════════════════════
# 2. SPECTRAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _calc_ispec(M, var_dens_2d, averaging=True, truncate=True, nfactor=1):
    vd = np.copy(var_dens_2d)
    vd[..., 0]  /= 2
    vd[..., -1] /= 2

    ll = np.fft.fftfreq(M) * M
    kk = np.arange(0, M // 2 + 1)
    k2d, l2d = np.meshgrid(kk, ll)

    wv = np.sqrt(k2d**2 + l2d**2)
    ll_max = np.abs(ll).max()
    kk_max = kk.max()
    kmax = min(ll_max, kk_max) if truncate else np.sqrt(ll_max**2 + kk_max**2)

    dkr = np.sqrt(2) * nfactor
    kr = np.arange(0, kmax, dkr)
    phr = np.zeros(kr.size)

    for i in range(kr.size):
        hi = kr[i + 1] if i < kr.size - 1 else kr[i] + dkr
        mask = (wv >= kr[i]) & (wv < hi)
        if np.any(mask):
            phr[i] = (
                vd[mask].mean() * (kr[i] + dkr / 2) * np.pi
                if averaging else vd[mask].sum() / dkr
            )
            phr[i] *= 2

    return kr + dkr / 2, phr


def enstrophy_spectrum(q_2d):
    q = to_2d(q_2d)
    M = q.shape[-1]
    spec2d = np.abs(np.fft.rfft2(q)) ** 2 / M**2
    return _calc_ispec(M, spec2d)


def energy_spectrum(q_2d):
    q = to_2d(q_2d)
    M = q.shape[-1]
    qhat = np.fft.rfft2(q)
    ll = np.fft.fftfreq(M) * M
    kk = np.arange(0, M // 2 + 1)
    k2d, l2d = np.meshgrid(kk, ll)
    wv2 = k2d**2 + l2d**2
    wv2[0, 0] = 1.0
    energy2d = np.abs(qhat) ** 2 / M**2 / wv2
    energy2d[0, 0] = 0.0
    return _calc_ispec(M, energy2d)


# ═══════════════════════════════════════════════════════════════════
# 3. OPEN FORECAST + REFERENCE
# ═══════════════════════════════════════════════════════════════════

print("\nOpening forecast:")
print(" ", FORECAST_ZARR)
fc = xr.open_zarr(FORECAST_ZARR, consolidated=False, chunks="auto")
print(fc)

print("\nOpening reference/test data:")
print(" ", REFERENCE_ZARR)
ref = xr.open_zarr(REFERENCE_ZARR, consolidated=False, chunks="auto")
print(ref)

if VAR not in fc:
    raise KeyError(f"{VAR!r} not in forecast. Variables: {list(fc.data_vars)}")
if VAR not in ref:
    raise KeyError(f"{VAR!r} not in reference. Variables: {list(ref.data_vars)}")

q_fc = fc[VAR]
q_ref = ref[VAR]

required_fc_dims = {"init_time", "ensemble", "lead_time", "lev", "y", "x"}
missing = required_fc_dims.difference(q_fc.dims)
if missing:
    raise ValueError(f"Forecast {VAR!r} missing dims {missing}. Dims: {q_fc.dims}")

n_init = q_fc.sizes["init_time"]
n_ens  = q_fc.sizes["ensemble"]
n_lead = q_fc.sizes["lead_time"]
ny     = q_fc.sizes["y"]
nx     = q_fc.sizes["x"]

init_values = pd.to_datetime(q_fc["init_time"].values)
lead_values = q_fc["lead_time"].values
lead_days   = forecast_lead_days(lead_values)

print(f"\nForecast: init={n_init}, ens={n_ens}, lead={n_lead}, y={ny}, x={nx}")
print(f"Lead range: {lead_days[0]:.2f}–{lead_days[-1]:.2f} days")

time_dim_ref = infer_time_dim(q_ref)
print(f"Reference time dim: {time_dim_ref}")

# Snapshot lead indices — N_SNAP evenly spaced, always including last step
snap_indices = np.unique(
    np.concatenate([
        np.linspace(0, n_lead - 1, N_SNAP, dtype=int),
        [n_lead - 1],
    ])
)
snap_indices = sorted(set(snap_indices.tolist()))
print(f"\nSnapshot lead indices ({len(snap_indices)}): {snap_indices}")
print(f"Snapshot lead days: {[f'{lead_days[i]:.1f}' for i in snap_indices]}")


# ═══════════════════════════════════════════════════════════════════
# 4. STREAMING PASS — compute all metrics without holding full arrays
# ═══════════════════════════════════════════════════════════════════
# Memory layout:
#   rmse_all[n_init, n_lead], spread_all[n_init, n_lead]  — scalars per (ic,lt)
#   truth_snap[n_init, N_SNAP, ny, nx]                    — truth at snap lts only
#   pred_snap[n_init, N_SNAP, ny, nx]                     — ens-mean at snap lts
#   truth_mean_lt[n_lead], truth_std_lt[n_lead]           — for drift plot
#   pred_mean_lt[n_lead],  pred_std_lt[n_lead]
#   ens_snap_ic0[N_SNAP, n_ens, ny, nx]                   — full ens for IC0 snaps
#   rank histogram accumulators at last lead time

print("\nStreaming pass over (init_time × lead_time)...")

rmse_all   = np.zeros((n_init, n_lead), dtype=np.float32)
spread_all = np.zeros((n_init, n_lead), dtype=np.float32)

# Drift accumulators (running sums for mean/std over ICs and space)
truth_spatial_mean = np.zeros((n_init, n_lead), dtype=np.float32)
pred_spatial_mean  = np.zeros((n_init, n_lead), dtype=np.float32)
truth_spatial_std  = np.zeros((n_init, n_lead), dtype=np.float32)
pred_spatial_std   = np.zeros((n_init, n_lead), dtype=np.float32)

# Snapshot storage
n_snap = len(snap_indices)
snap_idx_set = set(snap_indices)
snap_pos = {lt: pos for pos, lt in enumerate(snap_indices)}

truth_snap = np.zeros((n_init, n_snap, ny, nx), dtype=np.float32)
pred_snap  = np.zeros((n_init, n_snap, ny, nx), dtype=np.float32)

# Spectral accumulators at final lead time
LT_SPEC = n_lead - 1
ens_true_list, erg_true_list = [], []
ens_pred_list, erg_pred_list = [], []
kr_ref = None

# Rank histogram at final lead
ranks_rh = []

# Valid times for IC0 (for labels)
valid_times_ic0 = []

for ic in range(n_init):
    init_time = pd.Timestamp(init_values[ic])
    print(f"  IC {ic+1}/{n_init}: {init_time}", flush=True)

    for lt in range(n_lead):
        lead_td    = as_timedelta64(lead_values[lt])
        valid_time = init_time + lead_td

        if ic == 0:
            valid_times_ic0.append(valid_time)

        # Load truth (single time step, 2D)
        truth = select_reference_at_valid_time(q_ref, valid_time, lev_idx=LEV_IDX)

        # Load ensemble for this (ic, lt) slice only — shape (n_ens, ny, nx)
        ens_slice = (
            q_fc.isel(init_time=ic, lead_time=lt, lev=LEV_IDX)
            .load()
            .values.astype(np.float32)
        )
        pred = ens_slice.mean(axis=0)
        pred_std_field = ens_slice.std(axis=0)

        rmse_all[ic, lt]   = rmse(pred, truth)
        spread_all[ic, lt] = float(pred_std_field.mean())

        truth_spatial_mean[ic, lt] = float(truth.mean())
        pred_spatial_mean[ic, lt]  = float(pred.mean())
        truth_spatial_std[ic, lt]  = float(truth.std())
        pred_spatial_std[ic, lt]   = float(pred.std())

        if lt in snap_idx_set:
            s = snap_pos[lt]
            truth_snap[ic, s] = truth
            pred_snap[ic, s]  = pred

        # Spectral diagnostics at final lead
        if lt == LT_SPEC:
            kr, ens_t = enstrophy_spectrum(truth)
            _, erg_t  = energy_spectrum(truth)
            _, ens_p  = enstrophy_spectrum(pred)
            _, erg_p  = energy_spectrum(pred)
            if kr_ref is None:
                kr_ref = kr
            ens_true_list.append(ens_t)
            erg_true_list.append(erg_t)
            ens_pred_list.append(ens_p)
            erg_pred_list.append(erg_p)

            # Rank histogram: observable = spatial mean
            true_val  = float(truth.mean())
            pred_vals = ens_slice.mean(axis=(-2, -1))
            ranks_rh.append(int((pred_vals < true_val).sum()))

print("Streaming pass complete.")

# Aggregate drift stats over ICs
truth_mean_lt = truth_spatial_mean.mean(axis=0)
pred_mean_lt  = pred_spatial_mean.mean(axis=0)
truth_std_lt  = truth_spatial_std.mean(axis=0)
pred_std_lt   = pred_spatial_std.mean(axis=0)

# Normalisation stats from snapshot truth fields only (REF_IC, all snaps)
plot_mean = float(truth_snap[REF_IC].mean())
plot_std  = float(truth_snap[REF_IC].std())
plot_std  = max(plot_std, 1e-12)
print(f"Plot normalisation: mean={plot_mean:.6g}, std={plot_std:.6g}")

# Metric summaries
rmse_mean   = rmse_all.mean(axis=0)
rmse_std    = rmse_all.std(axis=0)
spread_mean = spread_all.mean(axis=0)
spread_std  = spread_all.std(axis=0)
ratio_mean  = spread_mean / (rmse_mean + 1e-12)


# ═══════════════════════════════════════════════════════════════════
# 5. SECTION A — SNAPSHOTS (N_SNAP lead steps only)
# ═══════════════════════════════════════════════════════════════════

print("\n── Section A: Field snapshots ──")

truth_seq_plot = maybe_normalize(truth_snap[REF_IC], plot_mean, plot_std)
pred_seq_plot  = maybe_normalize(pred_snap[REF_IC],  plot_mean, plot_std)
diff_seq_plot  = pred_seq_plot - truth_seq_plot

vmax_f = float(np.percentile(np.abs(np.stack([truth_seq_plot, pred_seq_plot])), 99))
vmax_e = float(np.percentile(np.abs(diff_seq_plot), 99))
vmax_f = max(vmax_f, 1e-12)
vmax_e = max(vmax_e, 1e-12)

fig, axes = plt.subplots(
    n_snap, 3,
    figsize=(10, 2.8 * n_snap),
    dpi=120,
    gridspec_kw={"hspace": 0.25, "wspace": 0.08},
)
if n_snap == 1:
    axes = axes[np.newaxis, :]

for s, lt in enumerate(snap_indices):
    axes[s, 0].imshow(truth_seq_plot[s], cmap="RdBu_r", origin="lower", vmin=-vmax_f, vmax=vmax_f)
    axes[s, 1].imshow(pred_seq_plot[s],  cmap="RdBu_r", origin="lower", vmin=-vmax_f, vmax=vmax_f)
    im = axes[s, 2].imshow(diff_seq_plot[s], cmap="seismic", origin="lower", vmin=-vmax_e, vmax=vmax_e)

    for col in range(3):
        axes[s, col].axis("off")

    axes[s, 0].text(
        -0.25, 0.5, f"lead={lead_days[lt]:.1f}d",
        transform=axes[s, 0].transAxes,
        va="center", ha="right", fontsize=8,
    )
    axes[s, 2].set_title(f"RMSE={rmse_mean[lt]:.4g}", fontsize=7, pad=2)

axes[0, 0].set_title("TRUTH q", fontsize=9, color="navy")
axes[0, 1].set_title("FORECAST ENS MEAN q", fontsize=9, color="firebrick")
axes[0, 2].set_title("ERROR mean − truth", fontsize=9, color="saddlebrown")

label_unit = "q normalized" if PLOT_NORMALIZED else "q physical"
err_unit   = "error normalized" if PLOT_NORMALIZED else "error physical"

fig.colorbar(axes[0, 1].images[0], ax=axes[:, 0:2], fraction=0.025, pad=0.02).set_label(label_unit, fontsize=8)
fig.colorbar(im, ax=axes[:, 2], fraction=0.035, pad=0.02).set_label(err_unit, fontsize=8)

fig.suptitle(
    f"Section A — Real forecast verification snapshots\n"
    f"IC={REF_IC}, lev={LEV_IDX}; {n_snap} evenly-spaced lead steps",
    fontsize=10, y=1.02,
)
path_A = os.path.join(OUT_DIR, "A_snapshots_real_truth_fixed_scale.png")
plt.savefig(path_A, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", path_A)


# ═══════════════════════════════════════════════════════════════════
# 6. SECTION B — GIF (N_SNAP frames)
# ═══════════════════════════════════════════════════════════════════

print("\n── Section B: Trajectory GIF ──")

frame_dir = os.path.join(OUT_DIR, "_frames_tmp")
os.makedirs(frame_dir, exist_ok=True)
filenames = []

for s, lt in enumerate(snap_indices):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
    axs[0].imshow(truth_seq_plot[s], cmap="RdBu_r", origin="lower", vmin=-vmax_f, vmax=vmax_f)
    axs[1].imshow(pred_seq_plot[s],  cmap="RdBu_r", origin="lower", vmin=-vmax_f, vmax=vmax_f)
    im = axs[2].imshow(diff_seq_plot[s], cmap="seismic", origin="lower", vmin=-vmax_e, vmax=vmax_e)

    axs[0].set_title("TRUTH q")
    axs[1].set_title("ENS MEAN q")
    axs[2].set_title(f"ERROR, RMSE={rmse_mean[lt]:.4g}")
    for ax in axs:
        ax.axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    vt = valid_times_ic0[lt] if lt < len(valid_times_ic0) else "?"
    fig.suptitle(f"IC={REF_IC}; lead={lead_days[lt]:.1f}d; valid={vt}", y=0.98)

    fname = os.path.join(frame_dir, f"frame_{s:03d}.png")
    plt.savefig(fname, bbox_inches="tight")
    filenames.append(fname)
    plt.close(fig)

gif_path = os.path.join(OUT_DIR, "B_trajectory_real_truth.gif")
frames = [Image.open(f) for f in filenames]
frames[0].save(gif_path, format="GIF", append_images=frames[1:],
               save_all=True, duration=600, loop=0)
for f in filenames:
    os.remove(f)
os.rmdir(frame_dir)
print("Saved:", gif_path)


# ═══════════════════════════════════════════════════════════════════
# 7. SECTION C — RMSE VS LEAD TIME
# ═══════════════════════════════════════════════════════════════════

print("\n── Section C: RMSE vs lead ──")

fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
ax.plot(lead_days, rmse_mean, lw=1.5, label="Mean RMSE over ICs")
ax.fill_between(lead_days,
                np.maximum(rmse_mean - rmse_std, 0),
                rmse_mean + rmse_std,
                alpha=0.2, label="±1 std over ICs")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("RMSE(q) physical units")
ax.set_title("Section C — Forecast RMSE vs real test.zarr truth")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
path_C = os.path.join(OUT_DIR, "C_rmse_vs_lead_real_truth.png")
plt.savefig(path_C, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", path_C)


# ═══════════════════════════════════════════════════════════════════
# 8. SECTION D — SPREAD VS SKILL
# ═══════════════════════════════════════════════════════════════════

print("\n── Section D: Spread–skill overlay ──")

fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
ax.plot(lead_days, spread_mean, lw=1.5, marker="o", ms=3, label="Ensemble spread")
ax.fill_between(lead_days,
                np.maximum(spread_mean - spread_std, 0),
                spread_mean + spread_std, alpha=0.2)
ax.plot(lead_days, rmse_mean, lw=1.5, marker="s", ms=3, ls="--", label="RMSE vs truth")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("q physical units")
ax.set_title("Section D — Spread–skill\n(spread < RMSE → underdispersed)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
path_D = os.path.join(OUT_DIR, "D_spread_skill_real_truth.png")
plt.savefig(path_D, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", path_D)


# ═══════════════════════════════════════════════════════════════════
# 9. SECTION H — SPREAD / RMSE RATIO  ← new section
# ═══════════════════════════════════════════════════════════════════

print("\n── Section H: Spread/RMSE ratio ──")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

# Left: ratio vs lead time
ax = axes[0]
ax.plot(lead_days, ratio_mean, lw=2, color="steelblue", label="spread / RMSE")
ax.axhline(1.0, color="k", ls="--", lw=1.5, label="ideal (ratio = 1)")
ax.fill_between(lead_days, 0.9, 1.1, alpha=0.1, color="green", label="±10% of ideal")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("spread / RMSE")
ax.set_title("Section H — Spread/RMSE ratio vs lead time")
ax.set_ylim(0, max(2.5, float(ratio_mean.max()) * 1.15))
ax.grid(True, alpha=0.3)
ax.legend()

# Right: spread vs RMSE scatter (one point per IC per sampled lead)
# Sample every max(1, n_lead//50) lead steps to avoid overplotting
stride = max(1, n_lead // 50)
ax2 = axes[1]
sc = ax2.scatter(
    rmse_all[:, ::stride].ravel(),
    spread_all[:, ::stride].ravel(),
    c=np.tile(lead_days[::stride], (n_init, 1)).ravel(),
    cmap="viridis", alpha=0.4, s=6,
)
lim = max(float(rmse_all.max()), float(spread_all.max())) * 1.05
ax2.plot([0, lim], [0, lim], "k--", lw=1.5, label="perfect calibration")
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_xlabel("RMSE (physical units)")
ax2.set_ylabel("Ensemble spread (physical units)")
ax2.set_title("Spread vs RMSE scatter\n(colour = lead time in days)")
ax2.legend(fontsize=8)
fig.colorbar(sc, ax=ax2, label="Lead time (days)")

fig.suptitle("Section H — Spread/RMSE ratio", fontsize=11, y=1.02)
plt.tight_layout()
path_H = os.path.join(OUT_DIR, "H_spread_rmse_ratio.png")
plt.savefig(path_H, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", path_H)

# Print ratio table at a stride of lead_days for readability
print("\nSpread/RMSE ratio at selected lead times:")
print(f"{'lead (d)':>10} | {'RMSE':>10} | {'spread':>10} | {'ratio':>8}")
print("-" * 46)
report_stride = max(1, n_lead // 20)
for lt in range(0, n_lead, report_stride):
    print(f"{lead_days[lt]:10.2f} | {rmse_mean[lt]:10.6g} | {spread_mean[lt]:10.6g} | {ratio_mean[lt]:8.4f}")
# Always print last
lt = n_lead - 1
print(f"{lead_days[lt]:10.2f} | {rmse_mean[lt]:10.6g} | {spread_mean[lt]:10.6g} | {ratio_mean[lt]:8.4f}")


# ═══════════════════════════════════════════════════════════════════
# 10. SECTION E — SPECTRAL DIAGNOSTICS AT FINAL LEAD
# ═══════════════════════════════════════════════════════════════════

print("\n── Section E: Spectral diagnostics ──")

ens_true_arr = np.array(ens_true_list)
erg_true_arr = np.array(erg_true_list)
ens_pred_arr = np.array(ens_pred_list)
erg_pred_arr = np.array(erg_pred_list)

ens_true_mean = ens_true_arr.mean(0);  ens_true_std = ens_true_arr.std(0)
erg_true_mean = erg_true_arr.mean(0)
ens_pred_mean = ens_pred_arr.mean(0);  ens_pred_std = ens_pred_arr.std(0)
erg_pred_mean = erg_pred_arr.mean(0)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)

ax = axes[0]
ax.loglog(kr_ref, ens_true_mean, lw=2, label="Truth")
ax.fill_between(kr_ref, np.maximum(ens_true_mean - ens_true_std, 1e-30),
                ens_true_mean + ens_true_std, alpha=0.2)
ax.loglog(kr_ref, ens_pred_mean, lw=2, ls="--", label="Forecast ens mean")
ax.fill_between(kr_ref, np.maximum(ens_pred_mean - ens_pred_std, 1e-30),
                ens_pred_mean + ens_pred_std, alpha=0.2)
ax.set_xlabel("Isotropic wavenumber k")
ax.set_ylabel(r"$E_q(k)$")
ax.set_title("Enstrophy spectrum")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

ax = axes[1]
ax.loglog(kr_ref, erg_true_mean, lw=2, label="Truth")
ax.loglog(kr_ref, erg_pred_mean, lw=2, ls="--", label="Forecast ens mean")
ax.set_xlabel("Isotropic wavenumber k")
ax.set_ylabel(r"$E_u(k) \propto E_q(k)/k^2$")
ax.set_title("Energy spectrum")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

ax = axes[2]
ratio_ens = ens_pred_mean / (ens_true_mean + 1e-30)
ratio_erg = erg_pred_mean / (erg_true_mean + 1e-30)
ax.semilogx(kr_ref, ratio_ens, lw=2, label="Enstrophy ratio")
ax.semilogx(kr_ref, ratio_erg, lw=2, ls="--", label="Energy ratio")
ax.axhline(1.0, color="k", ls=":", lw=1.5)
ax.fill_between(kr_ref, 0.9, 1.1, alpha=0.1, label="±10%")
ax.set_xlabel("Isotropic wavenumber k")
ax.set_ylabel("Forecast / truth")
ax.set_title("Spectral bias")
ax.set_ylim(0, 3)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle(
    f"Section E — Spectra at lead={lead_days[LT_SPEC]:.1f} d, "
    f"averaged over {n_init} ICs", fontsize=10, y=1.02,
)
plt.tight_layout()
path_E = os.path.join(OUT_DIR, "E_spectral_diagnostics_real_truth.png")
plt.savefig(path_E, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", path_E)


# ═══════════════════════════════════════════════════════════════════
# 11. SECTION F — CLIMATOLOGICAL DRIFT
# ═══════════════════════════════════════════════════════════════════

print("\n── Section F: Mean/std drift ──")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

axes[0].plot(lead_days, truth_mean_lt, lw=2, marker="o", ms=2, label="Truth")
axes[0].plot(lead_days, pred_mean_lt,  lw=2, marker="s", ms=2, ls="--", label="Forecast ens mean")
axes[0].set_xlabel("Lead time (days)")
axes[0].set_ylabel("Spatial mean of q")
axes[0].set_title("Mean drift")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(lead_days, truth_std_lt, lw=2, marker="o", ms=2, label="Truth")
axes[1].plot(lead_days, pred_std_lt,  lw=2, marker="s", ms=2, ls="--", label="Forecast ens mean")
axes[1].set_xlabel("Lead time (days)")
axes[1].set_ylabel("Spatial std of q")
axes[1].set_title("Variance / amplitude drift")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle("Section F — Climatological drift against real truth", fontsize=10, y=1.02)
plt.tight_layout()
path_F = os.path.join(OUT_DIR, "F_climatological_drift_real_truth.png")
plt.savefig(path_F, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", path_F)


# ═══════════════════════════════════════════════════════════════════
# 12. SECTION G — RANK HISTOGRAM AT FINAL LEAD
# ═══════════════════════════════════════════════════════════════════

print("\n── Section G: Rank histogram ──")

from collections import Counter
ideal = 1.0 / (n_ens + 1)
rank_counts = Counter(ranks_rh)

fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
ax.hist(ranks_rh, bins=np.arange(n_ens + 2) - 0.5, density=True,
        edgecolor="white", lw=0.5, label="Observed rank frequencies")
ax.axhline(ideal, color="k", ls="--", lw=2, label=f"Ideal uniform={ideal:.3f}")
ax.set_xlabel(f"Rank of truth among {n_ens} ensemble members")
ax.set_ylabel("Relative frequency")
ax.set_title(
    f"Section G — Rank histogram, lead={lead_days[LT_SPEC]:.1f} d\n"
    "Flat=calibrated; U-shape=underdispersed; skew=bias"
)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
path_G = os.path.join(OUT_DIR, "G_rank_histogram_real_truth.png")
plt.savefig(path_G, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", path_G)


# ═══════════════════════════════════════════════════════════════════
# 13. SUMMARY
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("DIAGNOSTIC SUMMARY — REAL TRUTH VERIFICATION")
print("=" * 72)
print(f"Forecast zarr : {FORECAST_ZARR}")
print(f"Reference zarr: {REFERENCE_ZARR}")
print(f"Variable      : {VAR}")
print(f"Dims          : init={n_init}, ens={n_ens}, lead={n_lead}, y={ny}, x={nx}")
print(f"Lead range    : {lead_days[0]:.2f}–{lead_days[-1]:.2f} days")

print("\nRMSE / spread / ratio vs lead (sampled):")
print(f"{'lead (d)':>10} | {'RMSE':>10} | {'spread':>10} | {'ratio':>8}")
print("-" * 46)
for lt in range(0, n_lead, report_stride):
    print(f"{lead_days[lt]:10.2f} | {rmse_mean[lt]:10.6g} | {spread_mean[lt]:10.6g} | {ratio_mean[lt]:8.4f}")
lt = n_lead - 1
print(f"{lead_days[lt]:10.2f} | {rmse_mean[lt]:10.6g} | {spread_mean[lt]:10.6g} | {ratio_mean[lt]:8.4f}")

print("\nMean/std drift:")
print(f"  truth mean: {truth_mean_lt[0]:.6g} -> {truth_mean_lt[-1]:.6g}")
print(f"  pred  mean: {pred_mean_lt[0]:.6g}  -> {pred_mean_lt[-1]:.6g}")
print(f"  truth std : {truth_std_lt[0]:.6g}  -> {truth_std_lt[-1]:.6g}")
print(f"  pred  std : {pred_std_lt[0]:.6g}   -> {pred_std_lt[-1]:.6g}")

n_at_extremes = rank_counts[0] + rank_counts[n_ens]
frac_extremes = n_at_extremes / max(n_init, 1)
ideal_extreme = 2 / (n_ens + 1)
print(f"\nRank histogram final lead:")
print(f"  outside ensemble = {frac_extremes:.2%}, ideal ≈ {ideal_extreme:.2%}")
print(f"  rank=0: {rank_counts[0]}/{n_init}")
print(f"  rank={n_ens}: {rank_counts[n_ens]}/{n_init}")

print("\nFigures saved to:", OUT_DIR)
print("=" * 72)