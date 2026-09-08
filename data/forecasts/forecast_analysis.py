#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
forecast_analysis_corrected.py

Diagnostics for ocean_flow forecast.zarr, comparing the forecast against the
real reference trajectory in test.zarr.

Expected forecast structure:
    q(init_time, ensemble, lead_time, lev, y, x)

The forecast pipeline builds:
    init_times = pd.date_range(start=cfg.init_start, end=cfg.init_end, freq=cfg.init_freq)
    lead_times = pd.timedelta_range(start=cfg.step_freq, end=cfg.lead_time, freq=cfg.step_freq)

Therefore the valid verification time is:
    valid_time = init_time + lead_time

This script:
  A. plots truth / ensemble mean / error snapshots with fixed color scales
  B. makes a GIF for one initial condition
  C. plots RMSE vs lead time against real truth
  D. plots ensemble spread vs RMSE
  E. plots spectral diagnostics at the final lead time
  F. plots mean/std drift against the truth trajectory
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
   f"/lustre/fswork/projects/rech/wbg/ukv59en/flow-esm/"
    "surrogate-template/ocean_flow/data/forecasts/sqg_flow_matching_serious_long_forecast_ens8/forecast.zarr"
)

REFERENCE_ZARR = (
    "/lustre/fsn1/projects/rech/wbg/ukv59en/"
    "ocean_flow_data/data/test.zarr"
)

VAR = "q"
LEV_IDX = 0

OUT_DIR =f"sqg_flow_matching_serious_long_forecast_ens8"
os.makedirs(OUT_DIR, exist_ok=True)

# For visualisation only. Metrics are always computed in physical units.
PLOT_NORMALIZED = True

# Which IC to show in snapshot/GIF panels.
REF_IC = 0

# If exact valid-time matching fails, allow nearest-neighbour time matching?
# Keep False first. If your reference times differ by tiny encoding offsets, set True.
ALLOW_NEAREST_TIME = False


# ═══════════════════════════════════════════════════════════════════
# 1. BASIC HELPERS
# ═══════════════════════════════════════════════════════════════════

def to_2d(arr):
    """Squeeze an array to exactly 2D (y, x)."""
    a = np.squeeze(np.asarray(arr, dtype=np.float32))
    if a.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze, got shape {a.shape}")
    return a


def rmse(a, b):
    """Root mean square error in physical units."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def as_timedelta64(x):
    """Robust conversion of lead_time coordinate values to pandas Timedelta."""
    return pd.to_timedelta(x)


def infer_time_dim(da):
    """
    Find the time dimension in a reference xarray DataArray.

    We prefer common names. If none match, we look for a dimension whose
    coordinate has datetime64 dtype.
    """
    preferred = ["time", "init_time", "date", "valid_time"]
    for name in preferred:
        if name in da.dims:
            return name

    for dim in da.dims:
        if dim in da.coords and np.issubdtype(da[dim].dtype, np.datetime64):
            return dim

    raise ValueError(
        f"Could not infer time dimension for {da.name}. "
        f"Dims={da.dims}; coords={list(da.coords)}. "
        "You may need to edit infer_time_dim()."
    )


def infer_level_dim(da):
    """Find vertical level dimension, if present."""
    for name in ["lev", "level", "z", "depth"]:
        if name in da.dims:
            return name
    return None


def select_reference_at_valid_time(ref_da, valid_time, lev_idx=0):
    """
    Select the reference/truth field at a given valid_time.

    Returns a 2D numpy array (y, x).
    """
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
    """Convert forecast lead_time coordinate values to days for labels."""
    days = []
    for x in lead_values:
        td = pd.to_timedelta(x)
        days.append(td / pd.Timedelta(days=1))
    return np.asarray(days, dtype=float)


def get_forecast_field(q_fc, ic, lt, lev_idx=0, member=None, ensemble_mean=False):
    """
    Extract forecast field.

    If ensemble_mean=True: average over ensemble dimension.
    Else use a specific member.
    """
    if ensemble_mean:
        da = q_fc.isel(init_time=ic, lead_time=lt, lev=lev_idx).mean("ensemble")
    else:
        if member is None:
            raise ValueError("member must be provided unless ensemble_mean=True")
        da = q_fc.isel(init_time=ic, ensemble=member, lead_time=lt, lev=lev_idx)
    return to_2d(da.values)


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
    raise KeyError(f"{VAR!r} not found in forecast dataset. Variables: {list(fc.data_vars)}")
if VAR not in ref:
    raise KeyError(f"{VAR!r} not found in reference dataset. Variables: {list(ref.data_vars)}")

q_fc = fc[VAR]
q_ref = ref[VAR]

required_fc_dims = {"init_time", "ensemble", "lead_time", "lev", "y", "x"}
missing = required_fc_dims.difference(q_fc.dims)
if missing:
    raise ValueError(f"Forecast variable {VAR!r} is missing dims {missing}. Dims are {q_fc.dims}")

n_init = q_fc.sizes["init_time"]
n_ens = q_fc.sizes["ensemble"]
n_lead = q_fc.sizes["lead_time"]
ny = q_fc.sizes["y"]
nx = q_fc.sizes["x"]

init_values = pd.to_datetime(q_fc["init_time"].values)
lead_values = q_fc["lead_time"].values
lead_days = forecast_lead_days(lead_values)

print("\nForecast dimensions:")
print(f"  init_time : {n_init}")
print(f"  ensemble  : {n_ens}")
print(f"  lead_time : {n_lead}")
print(f"  y, x      : {ny}, {nx}")
print(f"  lead days : {lead_days}")

print("\nReference variable:")
print(q_ref)
time_dim_ref = infer_time_dim(q_ref)
print(f"  inferred reference time dimension: {time_dim_ref}")


# ═══════════════════════════════════════════════════════════════════
# 4. BUILD MATCHED TRUTH AND FORECAST ARRAYS
# ═══════════════════════════════════════════════════════════════════
# This is the central correction:
#   truth[ic, lt] = test.zarr q at valid_time = init_time[ic] + lead_time[lt]
#   pred[ic, lt]  = forecast ensemble mean at same init/lead
#
# We materialise only the matched fields needed for the diagnostics.

print("\nBuilding matched truth / forecast arrays...")

truth_all = np.zeros((n_init, n_lead, ny, nx), dtype=np.float32)
pred_mean_all = np.zeros((n_init, n_lead, ny, nx), dtype=np.float32)
spread_all = np.zeros((n_init, n_lead), dtype=np.float32)
rmse_all = np.zeros((n_init, n_lead), dtype=np.float32)

# Optional full ensemble for rank/spread diagnostics.
# Shape is modest for your current runs; if ensemble or IC count becomes huge,
# replace this with chunked computations.
ens_all = np.zeros((n_init, n_ens, n_lead, ny, nx), dtype=np.float32)

valid_times = []

for ic in range(n_init):
    init_time = pd.Timestamp(init_values[ic])
    valid_times_ic = []

    for lt in range(n_lead):
        lead_td = as_timedelta64(lead_values[lt])
        valid_time = init_time + lead_td
        valid_times_ic.append(valid_time)

        truth = select_reference_at_valid_time(q_ref, valid_time, lev_idx=LEV_IDX)
        ens_members = q_fc.isel(init_time=ic, lead_time=lt, lev=LEV_IDX).values.astype(np.float32)
        # ens_members shape: (ensemble, y, x)

        pred = ens_members.mean(axis=0)

        truth_all[ic, lt] = truth
        pred_mean_all[ic, lt] = pred
        ens_all[ic, :, lt] = ens_members
        spread_all[ic, lt] = ens_members.std(axis=0).mean()
        rmse_all[ic, lt] = rmse(pred, truth)

    valid_times.append(valid_times_ic)

print("Matched valid times for IC 0:")
for lt, vt in enumerate(valid_times[0]):
    print(f"  lead={lead_days[lt]:6.2f} d -> valid_time={vt}")

# Normalisation stats for plotting only, based on the real truth fields.
plot_mean = float(truth_all.mean())
plot_std = float(truth_all.std())
print(f"\nPlot normalisation from matched truth: mean={plot_mean:.6g}, std={plot_std:.6g}")


# ═══════════════════════════════════════════════════════════════════
# 5. SECTION A — SNAPSHOTS WITH FIXED COLOR SCALES
# ═══════════════════════════════════════════════════════════════════

print("\n── Section A: Field snapshots ──")

truth_seq = truth_all[REF_IC]
pred_seq = pred_mean_all[REF_IC]
diff_seq = pred_seq - truth_seq

truth_plot = maybe_normalize(truth_seq, plot_mean, plot_std)
pred_plot = maybe_normalize(pred_seq, plot_mean, plot_std)
diff_plot = pred_plot - truth_plot

vmax_f = float(np.percentile(np.abs(np.concatenate([
    truth_plot.reshape(-1),
    pred_plot.reshape(-1),
])), 99))

vmax_e = float(np.percentile(np.abs(diff_plot.reshape(-1)), 99))

# Avoid degenerate color limits.
vmax_f = max(vmax_f, 1e-12)
vmax_e = max(vmax_e, 1e-12)

n_rows = n_lead
fig, axes = plt.subplots(
    n_rows, 3,
    figsize=(10, 2.8 * n_rows),
    dpi=120,
    gridspec_kw={"hspace": 0.25, "wspace": 0.08},
)
if n_rows == 1:
    axes = axes[np.newaxis, :]

ims_f = []
ims_e = []

for lt in range(n_lead):
    im0 = axes[lt, 0].imshow(
        truth_plot[lt], cmap="RdBu_r", origin="lower",
        vmin=-vmax_f, vmax=vmax_f,
    )
    im1 = axes[lt, 1].imshow(
        pred_plot[lt], cmap="RdBu_r", origin="lower",
        vmin=-vmax_f, vmax=vmax_f,
    )
    im2 = axes[lt, 2].imshow(
        diff_plot[lt], cmap="seismic", origin="lower",
        vmin=-vmax_e, vmax=vmax_e,
    )

    ims_f.append(im1)
    ims_e.append(im2)

    for col in range(3):
        axes[lt, col].axis("off")

    axes[lt, 0].text(
        -0.25, 0.5, f"lead = {lead_days[lt]:.1f} d",
        transform=axes[lt, 0].transAxes,
        va="center", ha="right", fontsize=8,
    )

    axes[lt, 2].set_title(
        f"RMSE={rmse_all[REF_IC, lt]:.4g}",
        fontsize=7,
        pad=2,
    )

axes[0, 0].set_title("TRUTH q from test.zarr", fontsize=9, color="navy")
axes[0, 1].set_title("FORECAST ENS MEAN q", fontsize=9, color="firebrick")
axes[0, 2].set_title("ERROR mean − truth", fontsize=9, color="saddlebrown")

label_unit = "q normalized" if PLOT_NORMALIZED else "q physical"
err_unit = "error normalized" if PLOT_NORMALIZED else "error physical"

cbar_f = fig.colorbar(ims_f[0], ax=axes[:, 0:2], fraction=0.025, pad=0.02)
cbar_f.set_label(label_unit, fontsize=8)
cbar_f.ax.tick_params(labelsize=7)

cbar_e = fig.colorbar(ims_e[0], ax=axes[:, 2], fraction=0.035, pad=0.02)
cbar_e.set_label(err_unit, fontsize=8)
cbar_e.ax.tick_params(labelsize=7)

fig.suptitle(
    "Section A — Real forecast verification snapshots\n"
    f"IC={REF_IC}, lev={LEV_IDX}; fixed color scales across lead times",
    fontsize=10,
    y=1.02,
)

path_A = os.path.join(OUT_DIR, "A_snapshots_real_truth_fixed_scale.png")
plt.savefig(path_A, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", path_A)


# ═══════════════════════════════════════════════════════════════════
# 6. SECTION B — GIF FOR ONE IC
# ═══════════════════════════════════════════════════════════════════

print("\n── Section B: Trajectory GIF ──")

frame_dir = os.path.join(OUT_DIR, "_frames_tmp")
os.makedirs(frame_dir, exist_ok=True)
filenames = []

for lt in range(n_lead):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=100)

    axs[0].imshow(
        truth_plot[lt], cmap="RdBu_r", origin="lower",
        vmin=-vmax_f, vmax=vmax_f,
    )
    axs[1].imshow(
        pred_plot[lt], cmap="RdBu_r", origin="lower",
        vmin=-vmax_f, vmax=vmax_f,
    )
    im = axs[2].imshow(
        diff_plot[lt], cmap="seismic", origin="lower",
        vmin=-vmax_e, vmax=vmax_e,
    )

    axs[0].set_title("TRUTH q")
    axs[1].set_title("ENS MEAN q")
    axs[2].set_title(f"ERROR, RMSE={rmse_all[REF_IC, lt]:.4g}")

    for ax in axs:
        ax.axis("off")

    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"IC={REF_IC}; lead={lead_days[lt]:.1f} d; "
        f"valid={valid_times[REF_IC][lt]}",
        y=0.98,
    )

    fname = os.path.join(frame_dir, f"frame_{lt:03d}.png")
    plt.savefig(fname, bbox_inches="tight")
    filenames.append(fname)
    plt.close(fig)

gif_path = os.path.join(OUT_DIR, "B_trajectory_real_truth.gif")
frames = [Image.open(f) for f in filenames]
frames[0].save(
    gif_path,
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0,
)
for f in filenames:
    os.remove(f)
os.rmdir(frame_dir)

print("Saved:", gif_path)


# ═══════════════════════════════════════════════════════════════════
# 7. SECTION C — RMSE VS LEAD TIME AGAINST REAL TRUTH
# ═══════════════════════════════════════════════════════════════════

print("\n── Section C: RMSE vs lead ──")

rmse_mean = rmse_all.mean(axis=0)
rmse_std = rmse_all.std(axis=0)

fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
ax.plot(lead_days, rmse_mean, lw=2, marker="o", label="Mean over ICs")
ax.fill_between(
    lead_days,
    np.maximum(rmse_mean - rmse_std, 0.0),
    rmse_mean + rmse_std,
    alpha=0.2,
    label="±1 std over ICs",
)
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("RMSE(q), physical units")
ax.set_title("Section C — Forecast RMSE vs real test.zarr truth")
ax.grid(True, alpha=0.3)
ax.legend()

path_C = os.path.join(OUT_DIR, "C_rmse_vs_lead_real_truth.png")
plt.tight_layout()
plt.savefig(path_C, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", path_C)


# ═══════════════════════════════════════════════════════════════════
# 8. SECTION D — SPREAD VS SKILL
# ═══════════════════════════════════════════════════════════════════

print("\n── Section D: Spread–skill ──")

spread_mean = spread_all.mean(axis=0)
spread_std = spread_all.std(axis=0)

fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
ax.plot(lead_days, spread_mean, lw=2, marker="o", label="Ensemble spread")
ax.fill_between(
    lead_days,
    np.maximum(spread_mean - spread_std, 0.0),
    spread_mean + spread_std,
    alpha=0.2,
)

ax.plot(lead_days, rmse_mean, lw=2, marker="s", ls="--", label="RMSE vs real truth")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("q, physical units")
ax.set_title(
    "Section D — Spread–skill\n"
    "Spread < RMSE usually indicates underdispersion"
)
ax.grid(True, alpha=0.3)
ax.legend()

path_D = os.path.join(OUT_DIR, "D_spread_skill_real_truth.png")
plt.tight_layout()
plt.savefig(path_D, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", path_D)


# ═══════════════════════════════════════════════════════════════════
# 9. SECTION E — SPECTRAL DIAGNOSTICS AT FINAL LEAD
# ═══════════════════════════════════════════════════════════════════

print("\n── Section E: Spectral diagnostics ──")

LT_SPEC = n_lead - 1

ens_true_list, erg_true_list = [], []
ens_pred_list, erg_pred_list = [], []
kr_ref = None

for ic in range(n_init):
    q_true = truth_all[ic, LT_SPEC]
    q_pred = pred_mean_all[ic, LT_SPEC]

    kr, ens_t = enstrophy_spectrum(q_true)
    _, erg_t = energy_spectrum(q_true)
    _, ens_p = enstrophy_spectrum(q_pred)
    _, erg_p = energy_spectrum(q_pred)

    if kr_ref is None:
        kr_ref = kr

    ens_true_list.append(ens_t)
    erg_true_list.append(erg_t)
    ens_pred_list.append(ens_p)
    erg_pred_list.append(erg_p)

ens_true_arr = np.asarray(ens_true_list)
erg_true_arr = np.asarray(erg_true_list)
ens_pred_arr = np.asarray(ens_pred_list)
erg_pred_arr = np.asarray(erg_pred_list)

ens_true_mean = ens_true_arr.mean(axis=0)
erg_true_mean = erg_true_arr.mean(axis=0)
ens_pred_mean = ens_pred_arr.mean(axis=0)
erg_pred_mean = erg_pred_arr.mean(axis=0)

ens_true_std = ens_true_arr.std(axis=0)
ens_pred_std = ens_pred_arr.std(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)

ax = axes[0]
ax.loglog(kr_ref, ens_true_mean, lw=2, label="Truth")
ax.fill_between(
    kr_ref,
    np.maximum(ens_true_mean - ens_true_std, 1e-30),
    ens_true_mean + ens_true_std,
    alpha=0.2,
)
ax.loglog(kr_ref, ens_pred_mean, lw=2, ls="--", label="Forecast ens mean")
ax.fill_between(
    kr_ref,
    np.maximum(ens_pred_mean - ens_pred_std, 1e-30),
    ens_pred_mean + ens_pred_std,
    alpha=0.2,
)
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
    f"averaged over {n_init} ICs",
    fontsize=10,
    y=1.02,
)

path_E = os.path.join(OUT_DIR, "E_spectral_diagnostics_real_truth.png")
plt.tight_layout()
plt.savefig(path_E, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", path_E)


# ═══════════════════════════════════════════════════════════════════
# 10. SECTION F — CLIMATOLOGICAL DRIFT AGAINST TRUTH
# ═══════════════════════════════════════════════════════════════════

print("\n── Section F: Mean/std drift ──")

truth_mean = truth_all.mean(axis=(0, 2, 3))
pred_mean = pred_mean_all.mean(axis=(0, 2, 3))

truth_std = truth_all.std(axis=(2, 3)).mean(axis=0)
pred_std = pred_mean_all.std(axis=(2, 3)).mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

ax = axes[0]
ax.plot(lead_days, truth_mean, lw=2, marker="o", label="Truth")
ax.plot(lead_days, pred_mean, lw=2, marker="s", ls="--", label="Forecast ens mean")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("Spatial mean of q")
ax.set_title("Mean drift")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
ax.plot(lead_days, truth_std, lw=2, marker="o", label="Truth")
ax.plot(lead_days, pred_std, lw=2, marker="s", ls="--", label="Forecast ens mean")
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("Spatial std of q")
ax.set_title("Variance / amplitude drift")
ax.grid(True, alpha=0.3)
ax.legend()

fig.suptitle(
    "Section F — Climatological drift against real truth",
    fontsize=10,
    y=1.02,
)

path_F = os.path.join(OUT_DIR, "F_climatological_drift_real_truth.png")
plt.tight_layout()
plt.savefig(path_F, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", path_F)


# ═══════════════════════════════════════════════════════════════════
# 11. SECTION G — RANK HISTOGRAM USING REAL TRUTH
# ═══════════════════════════════════════════════════════════════════
# This uses spatial mean as the scalar observable. Later you can repeat with
# other observables: enstrophy, energy, selected Fourier modes, etc.

print("\n── Section G: Rank histogram ──")

LT_RH = n_lead - 1
ranks = []

for ic in range(n_init):
    true_val = float(truth_all[ic, LT_RH].mean())
    pred_vals = ens_all[ic, :, LT_RH].mean(axis=(-2, -1))
    ranks.append(int((pred_vals < true_val).sum()))

ideal = 1.0 / (n_ens + 1)
rank_counts = Counter(ranks)

fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
ax.hist(
    ranks,
    bins=np.arange(n_ens + 2) - 0.5,
    density=True,
    edgecolor="white",
    lw=0.5,
    label="Observed rank frequencies",
)
ax.axhline(ideal, color="k", ls="--", lw=2, label=f"Ideal uniform={ideal:.3f}")
ax.set_xlabel(f"Rank of truth among {n_ens} ensemble members")
ax.set_ylabel("Relative frequency")
ax.set_title(
    f"Section G — Rank histogram, lead={lead_days[LT_RH]:.1f} d\n"
    "Flat=calibrated; U-shape=underdispersed; skew=bias"
)
ax.grid(True, alpha=0.3)
ax.legend()

path_G = os.path.join(OUT_DIR, "G_rank_histogram_real_truth.png")
plt.tight_layout()
plt.savefig(path_G, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", path_G)


# ═══════════════════════════════════════════════════════════════════
# 12. SUMMARY
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("DIAGNOSTIC SUMMARY — REAL TRUTH VERIFICATION")
print("=" * 72)
print(f"Forecast zarr : {FORECAST_ZARR}")
print(f"Reference zarr: {REFERENCE_ZARR}")
print(f"Variable      : {VAR}")
print(f"Dims          : init={n_init}, ens={n_ens}, lead={n_lead}, y={ny}, x={nx}")
print(f"Lead days     : {lead_days}")
print()
print("RMSE vs lead, physical units:")
for lt in range(n_lead):
    print(
        f"  lead={lead_days[lt]:7.2f} d | "
        f"RMSE={rmse_mean[lt]:.6g} ± {rmse_std[lt]:.6g} | "
        f"spread={spread_mean[lt]:.6g} | "
        f"spread/RMSE={spread_mean[lt] / (rmse_mean[lt] + 1e-12):.3f}"
    )
print()
print("Mean/std drift:")
print(f"  truth mean: {truth_mean[0]:.6g} -> {truth_mean[-1]:.6g}")
print(f"  pred  mean: {pred_mean[0]:.6g} -> {pred_mean[-1]:.6g}")
print(f"  truth std : {truth_std[0]:.6g} -> {truth_std[-1]:.6g}")
print(f"  pred  std : {pred_std[0]:.6g} -> {pred_std[-1]:.6g}")
print()
n_at_extremes = rank_counts[0] + rank_counts[n_ens]
frac_extremes = n_at_extremes / max(n_init, 1)
ideal_extreme = 2 / (n_ens + 1)
print(f"Rank histogram final lead:")
print(f"  outside ensemble = {frac_extremes:.2%}, ideal ≈ {ideal_extreme:.2%}")
print(f"  rank=0: {rank_counts[0]}/{n_init}")
print(f"  rank={n_ens}: {rank_counts[n_ens]}/{n_init}")
print()
print("Figures saved to:", OUT_DIR)
print("=" * 72)
