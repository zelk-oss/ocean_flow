# CLAUDE.md

This is a surrogate modelling framework for geoscientific data with a clean separation between training and inference.

## Environment

Use the `ocean_flow` conda environment for all commands:

```bash
conda activate ocean_flow
```

Install the package and dependencies:
```bash
uv pip install -r requirements.txt
pip install -e .
```

Run python commands with:
```bash
conda run -n ocean_flow <python commmand>
```

## Commands

**Run the project's own tests (requires 100% coverage of `ocean_flow/`):**
```bash
python -m pytest -q
```

**Run the project's tests together with every pulled flow-* submodule's tests:**
```bash
flow-test
```

**Run a single test file:**
```bash
python -m pytest tests/test_training.py -v
```

**Run a single test by name:**
```bash
python -m pytest tests/test_training.py::TestFlowMatchingTrainingModuleFunctional::test_estimate_loss_uses_pipelines -v
```

**Train a model (Hydra-configured):**
```bash
flow-train
```

**Override Hydra config at runtime:**
```bash
flow-train batch_size=32 seed=42
```

**Run forecasting:**
```bash
flow-forecast ckpt_path=data/models/best.ckpt
```

**Build docs:**
```bash
cd docs && make html
```

## Architecture

Framework logic (data loading, pipelines, the abstract training/inference
base classes, the forecast runner) lives in the `flow-core`/`flow-train`/
`flow-forecast`/`flow-assimilate` git submodules. `ocean_flow/` itself only
holds the model-specific code: `networks/`, `modules/`, and `configs/`.

### Framework submodules
- `flow-core`: `PrePipeline`/`PostPipeline` and their component modules
  (`PreNormalization`, `TendencyPrediction`, `BoundingModule`, ...);
  `InferenceModule` abstract base; output-store/checkpoint/environment
  helpers for inference.
- `flow-train`: `TrainDataset` (zarr-backed) and `TrainDataModule`;
  `TrainingModule` abstract base (optimizer, EMA, train/val loop).
- `flow-forecast`: `ForecastModel` (stateful autoregressive rollout under
  `lightning.fabric.Fabric`), the forecast runner, config/input/output/
  restart/validation helpers.
- `flow-assimilate`: data-assimilation extension (added, not yet used by
  this project).

### Module layer (`ocean_flow/modules/`)
Project-owned subclasses of the framework base classes, implementing the
flow-matching model:
- `training.py` — `FlowMatchingTrainingModule(flow_train.TrainingModule)`:
  implements `estimate_loss()` (flow-matching velocity-field regression in
  latent/residual space).
- `forecast.py` — `FlowMatchingForecastModule(flow_core.inference.InferenceModule)`:
  implements `forward()` (stochastic one-step forecast by integrating the
  learned flow from noise to a residual, added to the current state).
- `data.py` — `FlowMatchingTrainDataModule(flow_train.TrainDataModule)`:
  wraps `TrainDataset` samples with `input`/`residual` keys expected by
  `FlowMatchingTrainingModule.estimate_loss()`.

### Network layer (`ocean_flow/networks/`)
Contains the neural network definition (`unet.py`).

### Configuration (`configs/`)
Hydra configuration with composable defaults. Key top-level configs:
- `train.yaml`: references `data`, `pre_pipeline`/`post_pipeline`,
  `network`, `train_module`, `callbacks`, `logger` sub-configs.
- `forecast.yaml`: references the same data/pipeline/network, plus
  `inference_module`, `io`, `dask` sub-configs. Key fields: `ckpt_path`,
  `init_start/end/freq`, `lead_time`, `step_freq`.

### Training flow
`flow-train` CLI → Hydra composes config → instantiates
`FlowMatchingTrainDataModule` + `FlowMatchingTrainingModule` (network +
pipelines + EMA) → `pl.Trainer.fit()`. EMA weights are updated each batch;
a final BN update happens at training end.

### Forecast flow
`flow-forecast` CLI → sets up Fabric + Dask client → loads `ForecastModel`
wrapping a `FlowMatchingForecastModule` on the main process → iterates
forecast-config batches → each batch: `InputReader` (Dask futures,
prefetched) → `ForecastModel.advance()` on main process → `OutputWriter`
(Dask writes) → waits for all writes before exit.

## Python Coding Style

All code must follow the project style (enforced in CI via tests):
- **Line length**: strictly < 80 characters
- **Docstrings**: NumPy style with `r'''` raw strings on all functions and classes; include Parameters, Returns, Raises, Examples sections
- **Type hints**: required on all functions
- **Imports**: three groups — System, External, Internal — each separated by a blank line
- **File header**: shebang + encoding + author comment block
- **Private helpers**: extracted with `_` prefix, named `_validate_*`, `_load_*`, `_prepare_*`, etc.
- Constants: `UPPER_SNAKE_CASE`; classes: `PascalCase`; functions/vars: `snake_case`
