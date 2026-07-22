#!/usr/bin/env bash
set -euo pipefail

cargo llvm-cov clean --workspace
source <(cargo llvm-cov show-env --sh)

cargo test --workspace --all-features --locked
if python -m pip --version >/dev/null 2>&1; then
    maturin develop --locked
else
    maturin develop --uv --locked
fi
pytest tests/ -v

cargo llvm-cov report --summary-only
