#!/bin/zsh
set -e
S=${0:a:h}
[ -d $S/venv-port ] || uv venv -q --python 3.12 $S/venv-port
rm -rf $S/wheels-port
( cd $S/port && CARGO_TARGET_DIR=$S/target-port maturin build --release -q -o $S/wheels-port -i $S/venv-port/bin/python 2>&1 | grep -E "^error|warning: unused|Built wheel" | tail -5 )
uv pip install -q --reinstall -p $S/venv-port/bin/python $S/wheels-port/*.whl numpy
echo "[build] port $(git -C $S/port rev-parse --short HEAD) $(date +%T)"
