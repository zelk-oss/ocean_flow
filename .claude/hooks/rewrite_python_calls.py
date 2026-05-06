#!/usr/bin/env python
# -*- coding: utf-8 -*-

r'''Rewrite Bash python calls to use the project conda environment.

The hook reads Claude Code's PreToolUse JSON input from stdin, rewrites direct
Python command segments to run through ``conda run -n
ocean_flow``,
and returns the updated Bash command when a rewrite is needed.

Examples
--------
>>> _rewrite_command('cd /tmp && python script.py', 'example')
'cd /tmp && conda run -n example python script.py'

>>> _rewrite_command('echo hi', 'example') is None
True

>>> _rewrite_command(
...     'cd tmp/ && conda run -n example python script.py',
...     'example',
... ) is None
True
'''

import json
import re
import sys
from typing import Optional


COMMAND_BOUNDARY = r'(^|(?:&&|\|\||[;|])\s*)'
PYTHON_TOKEN = r'python(?:\d+(?:\.\d+)*)?'


def _already_wrapped(command: str, env_name: str) -> bool:
    r'''Return whether the command already uses the target wrapper.

    Parameters
    ----------
    command : str
        Shell command to inspect.
    env_name : str
        Conda environment name.

    Returns
    -------
    bool
        ``True`` when the target conda wrapper is already present.
    '''

    pattern = re.compile(
        rf'{COMMAND_BOUNDARY}conda\s+run\s+-n\s+'
        rf'{re.escape(env_name)}\s+{PYTHON_TOKEN}\b'
    )
    return pattern.search(command) is not None


def _rewrite_command(command: str, env_name: str) -> Optional[str]:
    r'''Rewrite direct python command segments to use ``conda run``.

    Parameters
    ----------
    command : str
        Bash command string supplied by Claude Code.
    env_name : str
        Conda environment name.

    Returns
    -------
    Optional[str]
        The rewritten command, or ``None`` when no rewrite is required.
    '''

    if _already_wrapped(command, env_name):
        return None

    pattern = re.compile(
        rf'{COMMAND_BOUNDARY}'
        r'(?P<assignments>(?:[A-Za-z_][A-Za-z0-9_]*='
        r'(?:"[^"]*"|\'[^\']*\'|[^ \t;&|]+)\s+)*)'
        rf'(?P<python>{PYTHON_TOKEN})\b'
    )

    def _replace(match: re.Match[str]) -> str:
        return (
            f'{match.group(1)}'
            f'{match.group("assignments")}'
            f'conda run -n {env_name} {match.group("python")}'
        )

    rewritten, count = pattern.subn(_replace, command)
    if count == 0:
        return None

    return rewritten


def main() -> int:
    r'''Run the hook and emit the updated Bash command when needed.

    Returns
    -------
    int
        Process exit status.
    '''

    data = json.load(sys.stdin)
    command = data.get('tool_input', {}).get('command', '')
    if not isinstance(command, str):
        return 0

    updated_command = _rewrite_command(
        command,
        'ocean_flow',
    )
    if updated_command is None:
        return 0

    json.dump(
        {
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'allow',
                'updatedInput': {
                    'command': updated_command,
                },
            },
        },
        sys.stdout,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
