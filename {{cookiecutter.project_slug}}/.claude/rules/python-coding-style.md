---
paths:
  - "**/*.py"
applyTo: "**/*.py"
description: This file describes the Python coding style for the project
---
# Python Coding Style
**All agents must enforce these standards. This is non-negotiable.**

## Core Standards (Non-Negotiable)

| Element | Standard |
|---------|----------|
| **Line length** | < 80 characters (strictly) |
| **Naming** | PascalCase (classes), snake_case (functions/vars), UPPER_SNAKE (constants), `_prefix` (private) |
| **Type hints** | Required on ALL functions: `def func(x: int) -> str:` |
| **Docstrings** | NumPy style on ALL functions/classes with Parameters, Returns, Raises, Examples |
| **Spacing** | 4-space indent, PEP 8, blank line between methods |

## File Structure (Mandatory Template)

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

r'''Module docstring: brief summary.'''

# System modules
import logging
from typing import Optional, List

# External modules
import torch
import numpy as np

# Internal modules
from src.pipelines import BasePipeline

main_logger = logging.getLogger(__name__)
__all__ = ['PublicClass', 'public_function']
```

## Type Hints & Docstrings (Mandatory)

```python
def load_data(
    filepath: str,
    normalize: bool = True,
) -> torch.Tensor:
    r'''
    Brief one-line summary (<80 chars).
    
    Longer description explaining purpose and behavior.
    
    Parameters
    ----------
    filepath : str
        Path to data file.
    normalize : bool, optional, default = True
        If True, apply normalization. If False, use raw data.
    
    Returns
    -------
    torch.Tensor
        Loaded data with shape (N, *).
    
    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If file format is unsupported.
    
    Examples
    --------
    >>> data = load_data('train.pt')
    >>> data.shape
    torch.Size([1000, 256])
    '''
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    # Implementation
```

## Multi-line Break Rules (Line Length < 80)

**Assignments with conditionals:**
```python
# ✓ Parenthesized ternary (3 lines)
self.variables = (
    list(user_vars)
    if user_vars is not None
    else []
)
```

**Method calls:**
```python
# ✓ Break after opening paren
result = function_with_long_name(
    param1, param2, param3,
    kwarg1=value1, kwarg2=value2
)
```

**Long error messages:**
```python
raise ValueError(
    f"Variable '{var}' has dims {actual}, "
    f"expected {expected}"
)
```

## Helper Method Extraction (Readability & Testing)

When `__init__` or methods grow complex, extract into focused private helpers:

```python
class DataModule:
    def __init__(self, path: str, config: Dict):
        self._validate_inputs(path, config)  # Extract early
        self.data = self._load_data(path)
        self._config = self._prepare_config(config)
    
    def _validate_inputs(self, path: str, config: Dict) -> None:
        r'''
        Validate path and config exist.
        '''
        if not Path(path).exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if "required_key" not in config:
            raise ValueError("Config missing 'required_key'")
    
    def _load_data(self, path: str) -> xr.Dataset:
        r'''
        Load and return dataset.
        '''
        return xr.open_dataset(path)
    
    def _prepare_config(self, config: Dict) -> Dict:
        r'''
        Normalize and return config.
        '''
        return {k: v for k, v in config.items() if v is not None}
```

**Pattern**: Name with action prefix (`_validate_*`, `_load_*`, `_detect_*`, 
`_prepare_*`), one responsibility each.

## Class Docstrings: Include Attributes

```python
class Dataset:
    r'''
    Base class for data loading.
    
    This class handles zarr file I/O and ensemble dimension
    detection.
    
    Parameters
    ----------
    data_path : str
        Path to zarr dataset.
    
    Attributes
    ----------
    ensemble_size : int
        Number of ensemble members. Defaults to 1 if not
        present in data.
    n_times : int
        Number of time steps in dataset.
    '''
```

## Error Handling: Fail Fast & Clearly

```python
def process(self, data: xr.Dataset, schema: Dict) -> torch.Tensor:
    r'''
    Validate, load, and process data.
    '''
    
    # Fail fast on type/existence checks
    if not isinstance(data, xr.Dataset):
        raise TypeError(f"Expected Dataset, got {type(data)}")
    
    # Validate schema requirements
    for var, dims in schema.items():
        if var not in data.data_vars:
            raise ValueError(
                f"Missing variable '{var}'. "
                f"Found: {list(data.data_vars)}"
            )
    
    # Now proceed with complex logic
    return self._convert(data)
```

## Conditional Logic Simplification

**Before (redundant):**
```python
if aux_path is not None and aux_vars is None:
    logger.warning("Path provided, no vars")
elif aux_path is None and aux_vars is not None:
    logger.warning("Vars provided, no path")
elif aux_path is not None and aux_vars is not None:
    self._load_auxiliary(aux_path, aux_vars)
```

**After (clear & minimal):**
```python
if aux_path is not None and aux_vars is not None:
    self._load_auxiliary(aux_path, aux_vars)
elif aux_path is not None:
    logger.warning("Path provided but no variables")
elif aux_vars is not None:
    logger.warning("Variables specified but no path")
```

## Comments: Minimal & Precise

```python
# ✓ GOOD - Explains WHY
# Use zarr to avoid multithreading/dask issues
self.dataset = zarr.open_group(path)

# ✗ UNNECESSARY - Code is clear
# Set batch size to 32
batch_size = 32
```

## Validation Pipeline Pattern

When multiple validation steps needed, centralize them:

```python
def __init__(self, vars: List[str], aux_path: Optional[str] = None):
    self.variables = vars
    self._validate_variables()          # Check vars exist
    self.ensemble_size = self._detect_ensemble_size()
    self._load_auxiliary(aux_path)      # Load + validate
    self._validate_aux_ensemble()       # Check dimensions match
    
def _validate_variables(self) -> None:
    r'''
    Ensure all required variables exist in dataset.
    '''
    for var in self.variables:
        if var not in self.dataset:
            raise ValueError(f"Variable '{var}' not found")

def _validate_aux_ensemble(self) -> None:
    r'''
    Ensure auxiliary ensemble size matches dataset.
    '''
    if self.ensemble_size <= 1:
        return  # No validation needed
    for var in self.auxiliary_variables:
        if var_shape[0] != self.ensemble_size:
            raise ValueError(f"Aux ensemble mismatch: {var}")
```

## Quick Checklist

- [ ] Max line 80 chars (break long assignments, calls, messages)
- [ ] Type hints on ALL functions
- [ ] Docstrings with Parameters, Returns, Raises, See Also, Examples
- [ ] Private helpers extracted with `_` prefix for complexity
- [ ] Validation logic centralized in `_validate_*` methods
- [ ] Optional dimensions handled with safe defaults
- [ ] Error messages split across lines when long
- [ ] Comments only explain non-obvious logic
- [ ] File header + imports organized
- [ ] `__all__` defined
