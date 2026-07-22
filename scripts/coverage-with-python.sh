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

# Floor set just below the measured total (94.99% lines); only lower it
# deliberately, never raise it above what's actually measured.
cargo llvm-cov report --summary-only --fail-under-lines 94
