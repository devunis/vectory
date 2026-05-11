"""Run Vectory quality checks and optional staged-file auto-fixes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def staged_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def existing_python_files(paths: list[str]) -> list[str]:
    return [path for path in paths if path.endswith(".py") and (ROOT / path).exists()]


def restage(paths: list[str]) -> int:
    status = 0
    for path in paths:
        if not (ROOT / path).exists():
            continue
        status |= run(["git", "add", "--", path])
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true", help="Only auto-fix staged Python files")
    parser.add_argument(
        "--restage", action="store_true", help="Restage files changed by auto-fixes"
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest")
    args = parser.parse_args()

    tracked = staged_files() if args.staged else []
    lint_targets = (
        existing_python_files(tracked) if args.staged else ["vectory", "tests", "scripts"]
    )

    status = 0
    if lint_targets:
        status |= run([sys.executable, "-m", "ruff", "check", "--fix", *lint_targets])
        status |= run([sys.executable, "-m", "ruff", "format", *lint_targets])
        if args.restage:
            status |= restage(tracked)

    if not args.skip_tests:
        status |= run([sys.executable, "-m", "pytest", "tests/", "-q"])

    return status


if __name__ == "__main__":
    raise SystemExit(main())
