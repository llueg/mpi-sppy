#!/bin/bash

#SOLVER=xpress_persistent
SOLVER=gurobi_direct

mpiexec --oversubscribe -n 3 python netdes_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=1000.0 --instance-name=network-30-10-H-05 --max-iterations=100 --rel-gap=-0.01 --lagrangian --xhatshuffle --intra-hub-conv-thresh=-0.1 --max-stalled-iters=700  #--ph-track-progress --track-convergence 1 --track-xbars 1 --track-nonants 1 --track-duals 1 --tracking-folder="results/network-10-20-H-05/l1-rho1e3/" #--tee-rank0-solves
#116823.8200 