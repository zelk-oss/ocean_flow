---
name: forecast-model
description: |
  Run model forecasting for this repository using the `scripts/forecast.py` entrypoint.
  This skill launches forecasting under the `{{cookiecutter.project_slug}}` conda environment and
  keeps checking until the forecasting process exits.

  Trigger this skill when the user asks for a forecast run, a forecasting experiment, or
  when they share a command like `python scripts/forecast.py ...` and want it executed.
---

# What this skill does

This skill is responsible for:

- Running the forecasting entrypoint `scripts/forecast.py` from the repository root.
- Ensuring the run happens inside the `{{cookiecutter.project_slug}}` conda environment.
- Monitoring the forecasting subprocess and periodically reporting whether it is still running.

It does **not** attempt to interpret or modify model configs; it simply runs the forecast
command the user provides (or a sensible default) and watches until completion.

# When to use

Use this skill when the user wants to:

- Run a forecasting experiment (e.g., `python scripts/forecast.py model=era5 ...`).
- Start a new forecast and see live progress until completion.
- Confirm a given forecast command completes successfully (including early failures).

# How to run

1. **Ensure you are at the repository root.**
2. Run forecasting using the `{{cookiecutter.project_slug}}` conda environment.

The canonical command is:

```bash
cd code
conda run -n {{cookiecutter.project_slug}} --no-capture-output python scripts/forecast.py <hydra overrides...>
```

> Note: `conda run` avoids needing to `source activate` and works in non-interactive shells.

## Monitoring progress

To satisfy the “repeatedly check if the model forecasting is still running” requirement,
run the forecast command under a small monitor loop that prints a heartbeat every few seconds
while the process is alive.

This skill reuses the `run-python` skill’s monitor helper located at
`./.claude/skills/run-python/scripts/monitor_python.py`.

Example usage (from the repo root):

```bash
python .claude/skills/run-python/scripts/monitor_python.py \
  --env {{cookiecutter.project_slug}} \
  --cwd code \
  --check-interval 10 \
  -- python scripts/forecast.py model=era5 +experiments=era5/20260313_era5_posterior_crps \
      device=cuda:0 n_ens=16 n_forecast_steps=60
```

This will:
- activate the `{{cookiecutter.project_slug}}` environment via `conda run`
- start the forecasting process in ``
- print a heartbeat line every 10 seconds until forecast exits
- propagate the forecast process exit code

## Example (shallow water smoke test)

A minimal smoke test for the shallow water model (fast, small ensemble):

```bash
python .claude/skills/run-python/scripts/monitor_python.py \
  --env {{cookiecutter.project_slug}} \
  --cwd code \
  --check-interval 10 \
  -- python scripts/forecast.py model=shallow_water +experiments=shallow_water/det_mse \
      n_ens=2 n_forecast_steps=2 device=cpu
```

# Tips

- If `conda run` fails, confirm the `{{cookiecutter.project_slug}}` environment exists (see `environment.yml`).
- If forecasting hangs, inspect the log output for exceptions and ensure the dataset files are available.
- You can adjust `--check-interval` to print status more or less frequently.
