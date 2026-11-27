#!/bin/zsh
S=${0:a:h}
rules=(); for w in 2 3 4 6 8; do for t in 0.3 0.5 0.7 0.9; do rules+=("$w,$t"); done; done
v=("main:OL" "proto:OL" "port:OL:PROBE_COARSE=climb")
for r in $rules; do v+=("port:OL:PROBE_COARSE=climb PROBE_STALL1=$r PROBE_STALL2=$r"); done
for r in $rules; do v+=("port:OL:PROBE_COARSE=climb PROBE_STALL1=3,0.5 PROBE_STALL2=$r"); done
for r in $rules; do v+=("port:OL:PROBE_COARSE=climb PROBE_STALL1=$r PROBE_STALL2=3,0.5"); done
$S/iters.sh sweep "${v[@]}"
