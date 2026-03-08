#!/usr/bin/env bash
set -euo pipefail

cargo llvm-cov clean --workspace
source <(cargo llvm-cov show-env --sh)

cargo test --workspace --all-features
if python -m pip --version >/dev/null 2>&1; then
    maturin develop
else
    maturin develop --uv
fi
pytest tests/ -v

cargo llvm-cov report --summary-only
