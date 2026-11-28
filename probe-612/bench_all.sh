#!/bin/zsh
# #612 timed runs: interleaved rotated rounds, 5 below 4M rows and 3 at 4M and above.
cd /private/tmp/claude-501/-Users-kristof-Projects-numfaber/2d00e8bb-2c86-4177-b4c4-5533593f61d2/scratchpad/h
B=tools/bench.py; PY=python3
S=akm_sorting1e4_near_shared_year; M=akm_mobility0.001_near_shared_year
# Q1, Q2 (unsmoothed aggregates) on the sorted cell, every size
$PY $B sorted 5 3 "base:OL|PROBE_COARSE=climb;cand:OL|PROBE_COARSE=climb PROBE_SPACE=components-lift;cand:OL|PROBE_COARSE=climb PROBE_SPACE=agg PROBE_SA_THETA=0.01" /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${S}_250k.npz /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${S}_1000k.npz /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${S}_4000k.npz /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${S}_10000k.npz
# Q4 (rigid + small terms), every size; Q2's aggregates below 4M
$PY $B mobility 5 3 "base:OL|PROBE_COARSE=climb;cand:OL|PROBE_COARSE=climb PROBE_SPACE=rigid-small-lift" /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${M}_250k.npz /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${M}_1000k.npz /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${M}_4000k.npz /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${M}_10000k.npz
$PY $B mobility_agg 5 3 "base:OL|PROBE_COARSE=climb;cand:OL|PROBE_COARSE=climb PROBE_SPACE=agg PROBE_SA_THETA=0.01" /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${M}_250k.npz /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${M}_1000k.npz
# Q2 smoothed aggregation (assembled S), fixed config, below 4M
$PY $B sa 5 3 "base:OL|PROBE_COARSE=climb;cand:F_schwarz|PROBE_COARSE=fixed PROBE_SPACE=sa PROBE_SA_THETA=0.02" /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${S}_250k.npz
$PY $B sa 5 3 "base:OL|PROBE_COARSE=climb;cand:F_schwarz|PROBE_COARSE=fixed PROBE_SPACE=sa PROBE_SA_THETA=0.01" /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${S}_1000k.npz
# Q5 batch of 8 at 1M
$PY $B batch 5 3 "base:OL|PROBE_COARSE=climb PROBE_BATCH=8;cand:OL|PROBE_COARSE=climb PROBE_BATCH=8 PROBE_RECYCLE=30" /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/${S}_1000k.npz
# easy panels with every knob set: must match base
$PY $B easy 3 3 "base:OL|PROBE_COARSE=climb;cand:OL|PROBE_COARSE=climb PROBE_SPACE=agg PROBE_SA_THETA=0.01;cand:OL|PROBE_COARSE=climb PROBE_SPACE=rigid-small-lift" /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/akm_sorting1e4_intercepts_1000k.npz /private/tmp/claude-501/-Users-kristof-Projects-numfaber/41f52e1a-1ea8-421d-9b68-b7fbb7e81df5/scratchpad/screen/panels/akm_mobility1.0_near_shared_year_1000k.npz
echo DONE
