---
name: run-python
description: |
  Run an arbitrary Python script in this repository using the `ocean_flow` conda environment.
  This skill executes the provided python command and monitors it until completion.

  Trigger this skill when the user asks to run a Python script (e.g., `python scripts/my_script.py ...`)
  and wants to see live progress or ensure the run completes successfully.
---

# What this skill does

This skill is responsible for:

- Running a Python command from the repository root under `conda run -n ocean_flow`.
- Ensuring the command is executed in the correct working directory (default: ``).
- Monitoring the subprocess and periodically reporting whether it is still running.

It does **not** interpret or modify any of the script arguments; it simply runs what the user provides.

# When to use

Use this skill when the user wants to:

- Execute a Python script in the repo (e.g., `python scripts/train.py ...`, `python scripts/forecast.py ...`, or any other script).
- Run a custom Python script or helper where you want live progress and a clear exit status.
- Confirm a given python command completes successfully (including early failures).

# How to run

1. **Ensure you are at the repository root.**
2. Run the python command using the `ocean_flow` conda environment.

The canonical command is:

```bash
python .claude/skills/run-python/scripts/monitor_python.py \
  --env ocean_flow \
  --cwd code \
  --check-interval 10 \
  -- python <your_script>.py <args...>
```

> Note: `conda run` avoids needing to `source activate` and works in non-interactive shells.

## Monitoring progress

This helper prints a heartbeat every few seconds while the python process is alive.
It also exits with the same return code as the python process.

## Example (run training script)

```bash
python .claude/skills/run-python/scripts/monitor_python.py \
  --env ocean_flow \
  --cwd code \
  --check-interval 10 \
  -- python scripts/train.py model=shallow_water +experiments=shallow_water/det_mse n_epochs=1 device=cpu
```

# Tips

- If `conda run` fails, confirm the `ocean_flow` environment exists (see `environment.yml`).
- If the command hangs, inspect the script output to find exceptions and ensure any required data files exist.
- You can adjust `--check-interval` to print status more or less frequently.
