#!/bin/bash

SOLVER=gurobi_direct
#SOLVER=cplex

#mpiexec -n 3 python -u -m mpi4py sslp_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=5.0 --instance-name=sslp_15_45_10 --max-iterations=50 --rel-gap=0.0 --subgradient --xhatshuffle --intra-hub-conv-thresh=-0.1 --subgradient-rho-multiplier=1.0

mpiexec -n 3 python -u -m mpi4py sslp_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=1.0 --instance-name=sslp_5_25_50 --max-iterations=150 --rel-gap=0.0 --lagrangian --xhatshuffle --intra-hub-conv-thresh=-0.1 --ph-track-progress --track-convergence 1 --track-xbars 1 --track-nonants 1 --track-duals 1 --tracking-folder="results/sslp_5_25_50/l2-rho1/"

#mpiexec -n 2 python -u -m mpi4py sslp_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=5.0 --instance-name=sslp_15_45_10 --max-iterations=50 --rel-gap=0.0 --lagranger --xhatshuffle --intra-hub-conv-thresh=-0.1 