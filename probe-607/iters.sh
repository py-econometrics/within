#!/bin/zsh
# usage: iters.sh OUT "build:config:ENV=V ENV=V" ... ; one run per panel per variant, deterministic quantities only
S=${0:a:h}; name=$1; out=$S/$name.jsonl; shift
for p in $S/panels/*.npz; do
  for spec in "$@"; do
    b=${spec%%:*}; rest=${spec#*:}; c=${rest%%:*}; envs=${rest#*:}; [ "$envs" = "$rest" ] && envs=""
    line=$(env RAYON_NUM_THREADS=8 ${=envs} $S/venv-$b/bin/python -I $S/run_one.py $p $c 2>/dev/null | tail -1)
    echo "{\"variant\": \"$spec\", \"r\": ${line:-null}}" >> $out
    echo "$(date +%T) ${p:t:r} $spec $(echo $line | python3 -c 'import json,sys;r=json.loads(sys.stdin.read() or "null");print(r and (r["iterations"],r["converged"],"%.1e"%r["backward"]))')" >> $S/$name.log
  done
done
