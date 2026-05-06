---
paths:
  - "tests/**"
  - "**/test_*.py"
applyTo: "tests/**, **/test_*.py"
description: This file describes the Python testing style for the project.
---
# Python Testing Style
**All agents must enforce these standards when writing or reviewing test code.**

## File Header

```python
# -*- coding: utf-8 -*-
r'''Tests for src/<module>/<file>.py.

Brief description of what is being tested.
'''
```

## Import Order

```python
# System modules
import logging

# External modules
import numpy as np
import pytest
import torch
import xarray as xr

# Internal modules
from src.data.dataset import TrainDataset
from tests.conftest import _create_test_zarr, DummyNetwork
```

No blank lines within each group; one blank line between groups.

## Section Separators

Use horizontal rules to label each logical block:

```python
# -----------------------------------------------------------
# Helper classes
# -----------------------------------------------------------

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

# -----------------------------------------------------------
# Unit tests
# -----------------------------------------------------------

# -----------------------------------------------------------
# Error tests
# -----------------------------------------------------------

# -----------------------------------------------------------
# Edge cases
# -----------------------------------------------------------
```

## Test Class Organization

Split tests into four responsibility classes. Naming follows `Test<Module><Role>`:

| Class suffix | Purpose |
|---|---|
| `Functional` | End-to-end behavior, real I/O, full pipelines |
| `Unittest` | Isolated logic with stubs/mocks, no side effects |
| `Errors` | Error conditions — `pytest.raises`, wrong inputs |
| `EdgeCases` | Boundary values, empty inputs, singleton dims |

```python
class TestDatasetFunctional:
    r'''End-to-end tests for TrainDataset.'''

class TestDatasetUnittest:
    r'''Isolated unit tests for dataset internals.'''

class TestDatasetErrors:
    r'''Error condition tests for TrainDataset.'''

class TestDatasetEdgeCases:
    r'''Boundary condition tests for TrainDataset.'''
```

## Test Method Naming & Docstrings

- Pattern: `test_<component_or_action>_<expected_outcome>`
- Every method has a **one-line** `r'''...'''` docstring that reads as a fact

```python
def test_variables_property_returns_state_and_forcing(
        self,
        zarr_store: zarr.Group,
) -> None:
    r'''variables property returns state + forcing names.'''
    ...

def test_getitem_raises_on_out_of_bounds(self) -> None:
    r'''__getitem__ raises IndexError past dataset length.'''
    with pytest.raises(IndexError):
        ...
```

## Helper Naming Conventions

Private test helpers are prefixed with `_`. Use descriptive action prefixes:

| Prefix | Use |
|---|---|
| `_build_xxx` | Construct complex objects (modules, configs) |
| `_make_xxx` | Assemble input data (batches, tensors, configs) |
| `_create_xxx` | Create file-backed stores/datasets |
| `_XxxStub` / `_XxxRecorder` | Inline stub classes with minimal behavior |
| `_ProbeXxx` | Classes that expose internal state for inspection |

```python
def _build_forecast_module(
        n_fcst_steps: int,
        store_path: str | None,
        zarr_store: object,
) -> ForecastModule:
    r'''Return a deterministic ForecastModule for tests.'''
    ...

def _make_batch(
        raw_times: np.ndarray,
        batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    r'''Build deterministic forecast batch tensors.'''
    ...
```

## Inline Stub Classes

For true isolation in unit tests, define minimal stubs *inside* the test method:

```python
def test_v2_fallback(self) -> None:
    r'''Falls back to _ARRAY_DIMENSIONS for v2.'''

    class _Array:
        def __init__(self) -> None:
            self.attrs = {
                "_ARRAY_DIMENSIONS": ("time", "x"),
            }

    names = _get_dim_names(_Array())
    assert names == ["time", "x"]
```

When the stub is reused across tests in the same class, hoist it to the class level.

## Assertion Patterns

Use the **most specific** assertion tool available:

```python
# NumPy arrays
np.testing.assert_allclose(
    actual, expected, rtol=1e-5, atol=0,
)

# PyTorch tensors
torch.testing.assert_close(
    actual, expected, atol=1e-5, rtol=1e-5,
)

# Float scalar comparisons
assert value == pytest.approx(0.1, rel=1e-4)

# Shape checks
assert output.shape == (2, 24, 4, 8)

# Membership / set checks
assert "lead_time" in ds.coords
assert set(["init_time", "lead_time"]).issubset(ds.coords)

# Error conditions — include match= when feasible
with pytest.raises(ValueError, match="not separated"):
    split_wd_params(model)

# Multiple exception types
with pytest.raises((AssertionError, TypeError)):
    ...
```

Avoid `assert a == b` for array/tensor comparisons — use the dedicated tools above.

## Fixture Conventions (conftest.py)

Shared fixtures live in `tests/conftest.py` with full NumPy-style docstrings:

```python
@pytest.fixture()
def zarr_store(tmp_path: pathlib.Path) -> zarr.Group:
    r'''
    Create a test zarr store with synthetic data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    zarr.Group
        The opened zarr group at the created store path.
    '''
    path = str(tmp_path / "store.zarr")
    return _create_test_zarr(path)
```

**Factory fixtures** — return a callable when tests need parameterized data:

```python
@pytest.fixture()
def mocked_auxiliary_netcdf(
        tmp_path: pathlib.Path,
) -> Callable:
    r'''Factory fixture creating auxiliary netCDF files.'''
    counter = 0

    def _factory(
            n_ens: Optional[int] = None,
            n_lat: int = 4,
            n_lon: int = 8,
    ) -> str:
        nonlocal counter
        counter += 1
        path = str(tmp_path / f"aux_{counter}.nc")
        _create_test_auxiliary_netcdf(
            path, n_ens=n_ens, n_lat=n_lat, n_lon=n_lon,
        )
        return path

    return _factory
```

## Synthetic Data Conventions

- Use a **fixed seed** for reproducibility:
  `np.random.default_rng(seed=19921225)`
- Default spatial grid: `n_lat=4, n_lon=8` (small, fast)
- Default data type: `np.float32` / `torch.float32`
- Default ensemble absent unless explicitly needed
- Use `rng.normal()` for continuous, `rng.integers()` for discrete

```python
rng = np.random.default_rng(seed=19921225)
data = rng.normal(size=(n_times, n_vars, n_lat, n_lon))
data = data.astype(np.float32)
```

## Monkeypatching

Use `monkeypatch.setattr` to isolate external dependencies:

```python
def test_on_train_end_delegates(
        self,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    r'''on_train_end calls SWA BN update with trainer data.'''
    recorded: dict = {}

    def _fake_update_bn(
            train_dataloader: object,
            ema_network: object,
    ) -> None:
        recorded["train_dataloader"] = train_dataloader
        recorded["ema_network"] = ema_network

    monkeypatch.setattr(
        torch.optim.swa_utils,
        "update_bn",
        _fake_update_bn,
    )
    module.on_train_end()
    assert recorded["ema_network"] is module.ema_network
```

## Arrange-Act-Assert Structure

Keep the three phases visually distinct with blank lines. No AAA comment labels:

```python
def test_predict_step_writes_trajectory(
        self,
        tmp_path: object,
) -> None:
    r'''predict_step writes initial + AR trajectory.'''
    zarr_store = _create_test_zarr(
        str(tmp_path / "input.zarr"), n_times=4,
    )
    module = _build_forecast_module(n_fcst_steps=2, ...)

    module.predict_step(batch=batch, batch_idx=0)

    ds = xr.open_zarr(output_path, consolidated=True)
    np.testing.assert_allclose(
        ds.states_surface.values[0, :, 0, 0, 0],
        np.array([1.0, 2.0, 3.0]),
    )
```

## Quick Checklist

- [ ] File header: `# -*- coding: utf-8 -*-` + module docstring
- [ ] Imports in three groups: System / External / Internal
- [ ] Section separators between helper and test blocks
- [ ] Test classes split into Functional / Unittest / Errors / EdgeCases
- [ ] Test method names: `test_<component>_<expected_outcome>`
- [ ] Every test method has a one-line `r'''...'''` docstring
- [ ] Helpers prefixed with `_`, named with action prefix
- [ ] Inline stubs for isolation; shared helpers in conftest.py
- [ ] `np.testing.assert_allclose` / `torch.testing.assert_close` for arrays
- [ ] `pytest.raises` with `match=` where possible
- [ ] Fixed seed (`19921225`) for all random data
- [ ] `np.float32` / `torch.float32` as default dtype
- [ ] Max 80 chars per line (same as coding-style)
