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

**Run all tests (requires ≥99% coverage):**
```bash
python -m pytest -q
```

**Run a single test file:**
```bash
python -m pytest tests/test_dataset.py -v
```

**Run a single test by name:**
```bash
python -m pytest tests/test_dataset.py::TestTrainDataset::test_init -v
```

**Train a model (Hydra-configured):**
```bash
python scripts/train.py
```

**Override Hydra config at runtime:**
```bash
python scripts/train.py batch_size=32 seed=42
```

**Run forecasting:**
```bash
python scripts/forecast.py ckpt_path=data/models/best.ckpt
```

**Build docs:**
```bash
cd docs && make html
```

## Architecture

This is a surrogate modelling framework for geoscientific data with a clean separation between training and inference.

### Data layer (`ocean_flow/data/`)
- `TrainDataset`: Zarr-backed PyTorch `Dataset`. Loads state and forcing variables, optionally with ensemble and auxiliary (static) variables from a netCDF file. Samples are sequences of `n_steps` consecutive time steps. The `__getitem__` index encodes both time and ensemble member.
- `LightningDataModule` in `data_module.py` wraps the dataset for Lightning training.

### Pipeline layer (`ocean_flow/pipelines/`)
Two symmetric pipeline types form the normalization/transform chain:
- `PrePipeline(ModuleDict, PreModule)`: applies transformations sequentially on inputs (physical → latent). Contains modules like `PreNormalization`.
- `PostPipeline(ModuleDict, PostModule)`: applies transformations sequentially on outputs (latent → physical), plus a `to_latent()` method that inverts the chain (used during training to bring targets into latent space).
- Individual pipeline modules: `PreNormalization`, `PostNormalization`, `TendencyModule`, `BoundingModule`.

### Module layer (`ocean_flow/modules/`)
Split into two independent hierarchies:

**Training modules** (Lightning `LightningModule`):
- `TrainingModule`: abstract base. Handles optimizer (AdamW + cosine LR), EMA tracking (`ema_network`), and the train/val loop. Subclasses implement `estimate_loss()`.

**Forecast modules** (Lightning `LightningModule`, inference only):
- `ForecastModule`: abstract base holding network + pipelines, defines the single-step `forward()` interface.
- `ForecastModel`: stateful autoregressive wrapper (not a `nn.Module`). Runs on the main process under `lightning.fabric.Fabric`. Maintains rolling state, calls `ForecastModule.forward()` repeatedly, supports multi-GPU via Fabric strategies.

### Network layer (`ocean_flow/networks/`)
Contains the neural network definition.

### Forecast pipeline (`ocean_flow/forecast/`)
Handles the full inference workflow:
- `ForecastConfig` / `generate_forecast_configs`: generates batched forecast configurations from init times.
- `InputReader`: reads initial conditions, auxiliary, and forcing data from zarr stores.
- `OutputWriter`: writes forecast trajectories to a zarr output store.
- `runner` / `prefetch`: orchestrates forecast loop with Dask `client.submit` futures for parallel IO (prefetching states, forcings, fire-and-forget writes).
- `checkpoint` / `environment`: model loading helpers and Fabric/Dask client setup.

### Configuration (`configs/`)
Hydra configuration with composable defaults. Key top-level configs:
- `train.yaml`: references `data`, `pipelines`, `network`, `train_module`, `callbacks`, `logger` sub-configs.
- `forecast.yaml`: references same data/pipeline/network, plus `forecast_module`, `io`, `dask` sub-configs. Key fields: `ckpt_path`, `init_start/end/freq`, `lead_time`, `step_freq`.

### Training flow
`scripts/train.py` → Hydra composes config → instantiates `LightningDataModule` + `TrainingModule` (contains network + pipelines + EMA) → `pl.Trainer.fit()`. EMA weights are updated each batch; a final BN update happens at training end.

### Forecast flow
`scripts/forecast.py` → sets up Fabric + Dask client → loads `ForecastModel` on main process → iterates `ForecastConfig` batches → each batch: `InputReader` (Dask futures, prefetched) → `ForecastModel.advance()` on main process → `OutputWriter` (Dask fire-and-forget writes) → waits for all writes before exit.

## Python Coding Style

All code must follow the project style (enforced in CI via tests):
- **Line length**: strictly < 80 characters
- **Docstrings**: NumPy style with `r'''` raw strings on all functions and classes; include Parameters, Returns, Raises, Examples sections
- **Type hints**: required on all functions
- **Imports**: three groups — System, External, Internal — each separated by a blank line
- **File header**: shebang + encoding + author comment block
- **Private helpers**: extracted with `_` prefix, named `_validate_*`, `_load_*`, `_prepare_*`, etc.
- Constants: `UPPER_SNAKE_CASE`; classes: `PascalCase`; functions/vars: `snake_case`
