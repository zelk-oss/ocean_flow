# Surrogate Model Template

A cookiecutter template for rapid development of surrogates models and
data-driven emulators. The template provides a complete, modular framework,
supporting both single-GPU and distributed multi-GPU training and forecasting
via PyTorch Lightning.

## Architecture Overview

This template enforces a strict separation between training and forecasting:
- **Training Module** encapsulates the learning loop, loss computation,
    gradient updates, and EMA tracking. Subclasses implement `estimate_loss()`
    for model-specific training logic. Uses full `PyTorch Lightning` for
    scalable training on multiple GPUs and with mixed precision.
- **Forecast Module** orchestrates autoregressive
    prediction via `dask`, reading trained checkpoints and writing
    predictions directly to Zarr format. Subclasses implement `forward()` for
    model-specific forecasting logic. Uses PyTorch lightning `fabric` for
    lightweight yet consistent handling of multi-GPU and mixed precision
    forecasting.

Data flows through pre-processing and post-processing pipelines that map
between physical and latent spaces. Pre-processing standardises inputs
(e.g., normalisation by computed statistics). Post-processing scales
predictions back to physical variables, applies physical bounds, and optionally
applies tendency-based updates.

## Installation

Python 3.10+ with PyTorch 2.0+ is targetted. Begin by cloning the template:

```bash
cookiecutter git@github.com:tobifinn/surrogate-template.git
cd <your-project-name>
conda env create -f environment.yml
conda activate <your-project-name>
uv pip install -r requirements.txt
pip install -e .
```

## Examples using this template

There are several example repository that have used this template. If you want
that your repository is shown here, create a GitHub issue or contact me.

- [Beginner example](https://github.com/flow-esm/template-example-beginner)
  The beginner example instantiates a deterministic neural network to forecast
  several variables from the ERA5 dataset at a coarse 64x32 resolution.
- [Advanced example](https://github.com/flow-exm/template-example-advanced) 🔜
  The advanced example trains a generative flow model with a transformer
  architecture for weather forecasting based on surface and level data from the
  ERA5 reanalysis at a coarse 64x32 resolution.
- [GenSIM](https://github.com/flow-esm/gensim)
  🔜 The next version of the Generative Sea-Ice Model will be based on this
  template. GenSIM instantiates a transformer architecture with a censored
  generative flow for sea-ice forecasting and simulations at mesoscale
  resolutions.

## What to Customise During Implementation

After instantiating your project with cookiecutter, customise the 
following components to your problem:

1. **Data Loading**
    - Place your dataset in zarr format under `data/train_data` with
    `train.zarr` for training, `val.zarr` for validation, and
    `test.zarr` for testing
    - Create a new configuration file in `configs/data/` for your
    dataset

2. **Normalization and Bounds**
    - Create a pre-pipeline file in `configs/pipeline` for your
    dataset, e.g., specifying the normalisation of the input data.
    - Create a post-pipeline file in `configs/pipeline` for your dataset and
    problem, e.g., specifying that the prediction should
    be denormalised, added to the initial conditions, and clipped
    into physical bounds.

3. **Model Architecture**
    - Place your neural network architecture in
    `<your_project_name>/networks`. The architecture should define
    what happens in a single application of the neural network.
    - Create a new configuration file in `configs/networks/` for your
    neural network

4. **Training module**
    - Place a training module, inheriting from
    `<your_project_name>/modules/train_module.py`, under
    `<your_project_name>/modules/`. The module has to overwrite
    `estimate_loss()` and defines what happens in a single training
    step
    - Create a new configuration file in `configs/modules/` for your 
    training parameters.
    - Update `configs/train.yaml`, directing to your data config,
    pre- and post-pipeline, your neural network architecture and the 
    training module. Modify the number of workers and batch size to 
    your computational capacities

5. **Forecast module**
    - Place a forecasting module, inheriting from
    `<your_project_name>/modules/forecast_module.py`, under
    `<your_project_name>/modules/`. The module has to overwrite
    `forward()` and defines what happens in a single forecasting step
    - Create a new configuration file in `configs/modules/` for your 
    forecast model parameters, e.g., number of integration steps in
    a generative flow model
    - Update `configs/forecast.yaml`, directing to your data config 
    (only needed to populate variables in the forecast config), pre- 
    and post-pipeline, your neural network architecture and the 
    forecast module. Insert the time information to run the forecasts.
    Modify the number of workers and batch size to  your computational
    capacities.

## Data Format and Preparation

To stay compatible to the included `TrainDataset`, `TrainDataModule`, and the
`InputReader` for forecasting, the data should be stored in Zarr format with
time, (optional) ensemble, variable, and spatial dimensions. While state
variables are predicted, forcing variables are prior known before running the
forecast. Additional auxiliary variables can be specified if there are static
fields such as the mesh and land-sea mask.

The data can look like:
```
data/train_data/<dataset_name>/
├── train.zarr/
│   ├── states_surface       # (time, ensemble, var_surface, lat, lon)
│   ├── states_levels        # (time, ensemble, var_levels, level, lat, lon)
│   └── forcings             # (time, ensemble, var_forcing, lat, lon)
└── val.zarr/               # (same structure)
```
In the configuration for the data, one would set `states_surface` and
`states_levels` as state variables and `forcings` as forcing variables.

Ideally, the dataset is kept in physical and directly interpretable 
quantities.
The normalisation and denormalisation is handled by the pre- and 
post-pipeline.

## Software stack

The training scripts are based on [PyTorch](https://pytorch.org) in combination
with [PyTorch lightning](https://lightning.ai) for an efficient training system
that can scale to parallel training on multiple GPUs. The training is logged
via [WandB](https://wandb.ai/).

The forecasting is performed by instantiating a [Dask](https://dask.org) graph
which specifies how data is loaded, forecasting steps are chained, and
forecasts are stored. A stateful forecasting model uses
[lightning's Fabric](https://lightning.ai/docs/fabric) for scalable forecasting
pipelines. For the data loading and storage,
[Xarray](https://docs.xarray.dev) and [Zarr](https://zarr.dev) is used.

[Hydra](https://hydra.cc/) is used for hierarchical, composable configuration.
[configs/train.yaml](configs/train.yaml) and
[configs/forecast.yaml](configs/forecast.yaml) define the main training and
forcasting entry points. Subdirectories encapsulate further the composition
of data, train modules, forcasting modules, pre- and post-pipelines, neural
network architectures.

## Update cookiecutter instances

The cookiecutter template is under active development, and new features and
improvements are regularly added. To keep your project up to date with the
latest version of the template, you can use the included sync script
(recommended) or perform a manual sync.

### Scripted sync (recommended)

The `scripts/sync-template.sh` script automates the process of checking for
template updates and merging them into your project. It validates metadata,
shows you what changed, asks for confirmation, and handles the merge.

1. Check for available updates:

    ```bash
    bash scripts/sync-template.sh --check
    ```

2. If updates are available, merge them:

    ```bash
    bash scripts/sync-template.sh --merge
    ```

    The script will display the list of new commits and ask for confirmation
    before proceeding. If there are merge conflicts, resolve them manually,
    then commit.

3. After merging, review the changes and run your test suite to verify
   everything works correctly.

The sync script also keeps track of the last synced template version in
`.cookiecutter-metadata.json`, so subsequent syncs only pull in new changes.

### Manual sync

If you prefer to sync manually or the script is not available, you can
perform the following steps. Note that the scripted approach is preferred as
it also updates the stored template SHA for future syncs.

1. Fetch the latest template and identify the new commit:

    ```bash
    git remote add template <template-repo-url>  # if not already added
    git fetch template main
    ```

2. Create a temporary re-rendered copy of the template using the latest
   commit and your project's cookiecutter values:

    ```bash
    TEMPLATE_SHA=$(git rev-parse template/main)
    TMPDIR=$(mktemp -d)
    git archive "$TEMPLATE_SHA" | tar -x -C "$TMPDIR"
    cookiecutter "$TMPDIR" --no-input --output-dir "$TMPDIR/rendered" \
        project_name="<your-project-name>" \
        # ... add all your cookiecutter context values
    ```

3. Copy the rendered output into a temporary branch and merge:

    ```bash
    git checkout -b template-update
    rsync -a --exclude='.git' "$TMPDIR/rendered/<project_slug>/" .
    git add -A && git commit -m "Template update to $TEMPLATE_SHA"
    git checkout <your-branch>
    git merge template-update
    ```

4. Resolve any merge conflicts, review the changes, and run your test suite.

5. Update the `template_sha` field in `.cookiecutter-metadata.json` to the
   new SHA so that future syncs start from the correct point.

6. Clean up:

    ```bash
    git branch -d template-update
    rm -rf "$TMPDIR"
    ```

## Contributing

Contributions to the template are very welcome! If you have an idea for a new
feature, improvement, or bug fix, please open a GitHub issue or submit a pull
request. When contributing, please ensure that your code follows the existing
style and includes appropriate tests and documentation.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

Tobias Sebastian Finn — tobias.finn@enpc.fr
