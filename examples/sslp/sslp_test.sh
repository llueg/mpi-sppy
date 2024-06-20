#!/bin/bash

#SOLVER=xpress_persistent
#SOLVER=cplex
SOLVER=cbc

mpiexec -host=localhost -n 3 python -u -m mpi4py sslp_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --linearize-proximal-terms --default-rho=5.0 --instance-name=sslp_15_45_10 --max-iterations=50 --rel-gap=0.0 --subgradient --xhatshuffle --presolve --intra-hub-conv-thresh=-0.1 --subgradient-rho-multiplier=1.0
