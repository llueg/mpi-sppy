#!/bin/bash

#SOLVER=xpress_persistent
SOLVER=gurobi_persistent

mpiexec --oversubscribe -n 3 python netdes_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=3000.0 --instance-name=network-10-20-L-01 --max-iterations=100 --rel-gap=-0.01 --subgradient --xhatshuffle --intra-hub-conv-thresh=-0.1 --max-stalled-iters=700  --ph-track-progress --track-convergence 1 --track-xbars 1 --track-nonants 1 --track-duals 1 --tracking-folder="results/network-10-20-L-01/l1-rho3e3-sg/" #--tee-rank0-solves
#116823.8200 