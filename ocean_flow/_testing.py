#!/usr/bin/env python
# -*- coding: utf-8 -*-

r'''Console entry point that runs the full Flow-ESM test suite.

``flow-test`` runs the test suite of the generated project together with
the test suites of every framework submodule (``flow-*``) that has been
pulled into the project. All tests are executed in a single, aggregated
``pytest`` run with combined coverage across the project and submodule
packages; execution continues through failures and the aggregated exit
code is non-zero if any test fails.
'''

# System modules
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

__all__ = ['main']


def _project_root() -> Path:
    r'''
    Return the root directory of the generated project.

    The package is installed in editable mode inside the project, so the
    project root is the parent of the package directory that contains
    this module.

    Returns
    -------
    Path
        Absolute path to the project root directory.

    Examples
    --------
    >>> _project_root().is_dir()
    True
    '''
    return Path(__file__).resolve().parent.parent


def _discover_targets(root: Path) -> Tuple[List[str], List[str]]:
    r'''
    Discover test directories and coverage packages to run.

    The generated project's own ``tests`` directory is included when it
    exists. Each sibling ``flow-*`` submodule directory that contains a
    ``tests`` directory is added as well, using its importable package
    name (the directory name with hyphens replaced by underscores) as
    the coverage target.

    Parameters
    ----------
    root : Path
        Project root directory to scan.

    Returns
    -------
    test_paths : list of str
        Relative paths of the ``tests`` directories to run.
    cov_packages : list of str
        Importable package names to measure coverage for.

    Examples
    --------
    >>> paths, pkgs = _discover_targets(Path('.'))
    >>> isinstance(paths, list) and isinstance(pkgs, list)
    True
    '''
    test_paths: List[str] = []
    cov_packages: List[str] = []

    if (root / 'tests').is_dir():
        test_paths.append('tests')
        # The importable package is the directory holding this module,
        # which need not match the project root directory name.
        package = Path(__file__).resolve().parent.name
        cov_packages.append(package.replace('-', '_'))

    for submodule in sorted(root.glob('flow-*')):
        if not submodule.is_dir():
            continue
        if not (submodule / 'tests').is_dir():
            continue
        test_paths.append(f'{submodule.name}/tests')
        cov_packages.append(submodule.name.replace('-', '_'))

    return test_paths, cov_packages


def _build_command(
    test_paths: List[str],
    cov_packages: List[str],
) -> List[str]:
    r'''
    Build the aggregated ``pytest`` command line.

    Parameters
    ----------
    test_paths : list of str
        Relative paths of the ``tests`` directories to run.
    cov_packages : list of str
        Importable package names to measure coverage for.

    Returns
    -------
    list of str
        Argument vector for a single ``pytest`` invocation.

    Examples
    --------
    >>> _build_command(['tests'], ['my_project'])[:4]
    ['...', '-m', 'pytest', 'tests']
    '''
    cov_args = [f'--cov={pkg}' for pkg in cov_packages]
    # Report coverage but do not enforce a threshold: this overrides any
    # ``--cov-fail-under`` set in pyproject so users not following TDD are
    # not blocked by an incomplete-coverage failure.
    return [
        sys.executable, '-m', 'pytest',
        *test_paths,
        *cov_args,
        '--cov-report=term-missing',
        '--cov-fail-under=0',
    ]


def main() -> int:
    r'''
    Run the aggregated Flow-ESM test suite.

    Discovers the project and pulled-submodule test directories, then
    runs a single ``pytest`` invocation across all of them with combined
    coverage. Execution continues through failures; the returned exit
    code is non-zero if any test fails.

    Returns
    -------
    int
        Process exit code from ``pytest`` (0 on success, 1 when no test
        directories are found).

    Examples
    --------
    >>> isinstance(main(), int)  # doctest: +SKIP
    True
    '''
    root = _project_root()
    test_paths, cov_packages = _discover_targets(root)

    if not test_paths:
        print(
            'flow-test: no test directories found in the project or '
            'any pulled flow-* submodule.'
        )
        return 1

    command = _build_command(test_paths, cov_packages)
    print(
        f'flow-test: running {len(test_paths)} test suite(s) '
        f'from {root}'
    )
    result = subprocess.run(command, cwd=str(root))
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
