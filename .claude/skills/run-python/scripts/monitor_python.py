#!/usr/bin/env python3
"""Run a Python command under conda and print a heartbeat while it is running.

This helper is designed for use by the `run-python` skill.

Usage example (from repo root):

  python .claude/skills/run-python/scripts/monitor_python.py \
    --env ocean_flow \
    --cwd code \
    --check-interval 10 \
    -- python scripts/my_script.py arg1=foo

The script will:
- cd into `--cwd`
- launch `conda run -n <env> --no-capture-output <command ...>`
- poll every `--check-interval` seconds and print a heartbeat while the process is alive
- exit with the same return code as the python process
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a python command and watch it until completion.")
    parser.add_argument(
        "--env",
        default="ocean_flow",
        help="Name of the conda environment to use (default: ocean_flow)",
    )
    parser.add_argument(
        "--cwd",
        default="code",
        help="Directory to cd into before running the command (default: code)",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=10.0,
        help="Seconds between heartbeat messages (default: 10)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command to run (passed after --).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.command:
        print("ERROR: missing command to run. Provide the python command after --.")
        return 2

    # argparse.REMAINDER will include the leading "--" if provided, so strip it.
    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    repo_root = os.getcwd()
    workdir = os.path.join(repo_root, args.cwd)

    if not os.path.isdir(workdir):
        print(f"ERROR: working directory does not exist: {workdir}")
        return 3

    print(f"Running python command in: {workdir}")
    print("Command:", " ".join(command))

    # Construct the conda run invocation
    conda_cmd: List[str] = [
        "conda",
        "run",
        "-n",
        args.env,
        "--no-capture-output",
    ]

    full_cmd = conda_cmd + command

    proc = subprocess.Popen(
        full_cmd,
        cwd=workdir,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    start_ts = time.time()
    last_print = 0.0
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - start_ts
                print(f"Python process exited with code {ret} after {elapsed:.1f}s")
                return ret

            now = time.time()
            if now - last_print >= args.check_interval:
                elapsed = now - start_ts
                print(f"Python process still running ({elapsed:.1f}s elapsed; pid={proc.pid})")
                last_print = now

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("KeyboardInterrupt received; terminating python process...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
