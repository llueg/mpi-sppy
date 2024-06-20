#!/bin/bash

#SOLVER=xpress_persistent
#SOLVER=cplex
SOLVER=cbc

mpiexec -n 3 python netdes_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --linearize-proximal-terms --default-rho=10000.0 --instance-name=network-10-10-L-01 --max-iterations=100 --rel-gap=0.01 --lagrangian --xhatshuffle
