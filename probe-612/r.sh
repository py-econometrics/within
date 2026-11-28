#!/bin/zsh
# usage: r.sh base|cand panel_stem config [ENV=V ...] — one run, compact line
H=${0:a:h}; P=/private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels
v=$1; panel=$2; cfg=$3; shift 3
env RAYON_NUM_THREADS=${PROBE_THREADS:-8} "$@" $H/venv-$v/bin/python -I $H/tools/run_one.py $P/$panel.npz $cfg | python3 -c '
import json,sys; r=json.loads(sys.stdin.read()); 
print(r["panel"], r["config"], "it=",r["iterations"], "cons=%.2f solve=%.2f"%(r["construct"],r["solve"]), "setup=%.0fMB end=%.0fMB"%(r["rss_setup_mb"],r["rss_end_mb"]), "coarse=",r["coarse"], "cert=",r["certified"], r["x_sha"], ("ritz=%s"%r["ritz"]) if "ritz" in r else ""); [print("   ",l) for l in r.get("probe",[])]'
