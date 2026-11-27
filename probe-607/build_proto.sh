#!/bin/zsh
set -e
S=${0:a:h}
[ -d $S/venv-proto ] || uv venv -q --python 3.12 $S/venv-proto
rm -rf $S/wheels-proto
( cd $S/proto && CARGO_TARGET_DIR=$S/target-proto maturin build --release -q -o $S/wheels-proto -i $S/venv-proto/bin/python 2>&1 | grep -E "^error|warning: unused|Built wheel" | tail -5 )
uv pip install -q --reinstall -p $S/venv-proto/bin/python $S/wheels-proto/*.whl numpy
echo "[build] proto $(git -C $S/proto rev-parse --short HEAD) $(date +%T)"
