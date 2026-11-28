#!/bin/zsh
# usage: build.sh base|cand  — release wheel of that worktree into venv-<name>
set -e
H=${0:a:h}; S=${H:h}; name=$1
src=$S/$([ $name = base ] && echo base || echo p611)
[ -d $H/venv-$name ] || uv venv -q --python 3.12 $H/venv-$name
rm -rf $H/wheels-$name
( cd $src && CARGO_TARGET_DIR=$S/target-$name maturin build --release -q -o $H/wheels-$name -i $H/venv-$name/bin/python 2>&1 | grep -E "^error|Built wheel" | tail -5 )
uv pip install -q --reinstall -p $H/venv-$name/bin/python $H/wheels-$name/*.whl numpy scipy
echo "[build] $name $(git -C $src rev-parse --short HEAD)$(git -C $src diff --quiet || echo +dirty) $(date +%T)"
