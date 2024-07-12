#!/bin/bash

SOLVERNAME="cbc"

# taking defaults for most farmer args
# --reduced-costs --rc-fixer --rc-fix-fraction-iter0=0.2 --rc-fix-fraction-iterK=0.2 --rc-progressive-fix-fraction 
#mpiexec -np 3 python -m mpi4py farmer_cylinders.py  --linearize-proximal-terms --proximal-linearization-tolerance=1e-3 --num-scens 3 --max-iterations=50 --crops-mult=10 --tracking-folder="tracking_test/" --ph-track-progress --track-convergence=1 --track-nonants=1 --xhatshuffle --lagrangian --bundles-per-rank=0 --default-rho=1 --rel-gap=-1 --abs-gap=-1 --solver-name=${SOLVERNAME}
mpiexec -np 3 python -m mpi4py sizes_cylinders.py --ph-track-progress --track-xbars=1 --display-progress --linearize-proximal-terms --proximal-linearization-tolerance=1e-3 --num-scens 3 --max-iterations=20   --xhatshuffle --lagrangian --bundles-per-rank=0 --default-rho=1 --rel-gap=-1 --abs-gap=-1 --solver-name=${SOLVERNAME}
