# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import math
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
import numpy as np
from mpisppy.cylinders.spcommunicator import communicator_array
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.utils.sputils import is_persistent 
from mpisppy import MPI

class ReducedCostsSpoke(LagrangianOuterBound):

    # converger_spoke_char = 'R'
    # # TODO: set option
    # bound_tol = 1e-6
    # consensus_threshold = 1e-3
    #th = self.opt.options['threshold']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This doesn't seem to be available in the spokes
        #self.bound_tol = self.opt.options['rc_options']['bound_tol']
        self.bound_tol = 1e-6
        self.consensus_threshold = 1e-3
        self.converger_spoke_char = 'R'
        # TODO: Could give above options thorugh config, but is it important enough?
        #options = self.opt.options


    def make_windows(self):
        if not hasattr(self.opt, "local_scenarios"):
            raise RuntimeError("Provided SPBase object does not have local_scenarios attribute")

        if len(self.opt.local_scenarios) == 0:
            raise RuntimeError("Rank has zero local_scenarios")

        vbuflen = 2
        for s in self.opt.local_scenarios.values():
            vbuflen += len(s._mpisppy_data.nonant_indices)

        self.nonant_length = self.opt.nonant_length

        self._modeler_fixed_nonants = set()

        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants.add(ndn_i)
            break
        #print(f"self._modeler_fixed_nonants: {self._modeler_fixed_nonants}")

        self._make_windows(1 + self.nonant_length, vbuflen)
        self._locals = communicator_array(vbuflen)
        # over load the _bound attribute here
        # so the rest of the class works as expected
        # first float will be the bound we're sending
        # indices 1:-1 will be the reduced costs, and
        # the last index will be the serial number
        self._bound = communicator_array(1 + self.nonant_length)
        # print(f"nonant_length: {self.nonant_length}, integer_nonant_length: {self.integer_nonant_length}")

    @property
    def rc(self):
        return self._bound[1:1+self.nonant_length]

    @rc.setter
    def rc(self, vals):
        self._bound[1:1+self.nonant_length] = vals

    def lagrangian_prep(self):
        """
        same as base class, but relax the integer variables and
        attach the reduced cost suffix
        """
        verbose = self.opt.options['verbose']
        # Split up PH_Prep? Prox option is important for APH.
        # Seems like we shouldn't need the Lagrangian stuff, so attach_prox=False
        # Scenarios are created here
        self.opt.PH_Prep(attach_prox=False)
        self.opt._reenable_W()

        if self.opt._presolver is not None:
            # do this before we relax the integer variables
            self.opt._presolver.presolve()

        relax_integer_vars = pyo.TransformationFactory("core.relax_integer_vars")
        for s in self.opt.local_subproblems.values():
            relax_integer_vars.apply_to(s)
            s.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        self.opt._create_solvers(presolve=False)

    def lagrangian(self):
        bound = super().lagrangian()
        if bound is not None:
            self.extract_and_store_reduced_costs(bound)
        return bound

    def extract_and_store_reduced_costs(self, outer_bound):
        self.opt.Compute_Xbar()
        # NaN will signal that the x values do not agree in
        # every scenario, we can't extract an expected reduced
        # cost
        is_minimizing = self.opt.is_minimizing
        rc = np.zeros(self.nonant_length)

        for sub in self.opt.local_subproblems.values():
            if is_persistent(sub._solver_plugin):
                # TODO: only load nonant's RC
                # TODO: what happens with non-persistent solvers? 
                # - if rc is accepted as a model suffix by the solver (e.g. gurobi shell), it is loaded in postsolve
                # - if not, the solver should throiw an error
                # - direct solvers seem to behave the same as persistent solvers
                # GurobiDirect needs vars_to_load argument
                # XpressDirect loads for all vars by default
                vars_to_load = [x for sn in sub.scen_list for _, x in self.opt.local_scenarios[sn]._mpisppy_data.nonant_indices.items()]
                sub._solver_plugin.load_rc(vars_to_load=vars_to_load)
            # TODO: warning ?

            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    #print(f"{xvar.name}, rc: {pyo.value(sub.rc[xvar])}, val: {xvar.value}, lb: {xvar.lb}, ub: {xvar.ub} ")
                    # fixed by modeler
                    if ndn_i in self._modeler_fixed_nonants:
                        rc[ci] = np.nan
                        continue
                    # if ndn_i in self._integer_proved_fixed_nonants:
                    #     if xvar.value == xvar.lb:
                    #         rc[ci] = math.inf if is_minimizing else -math.inf
                    #     else:
                    #         rc[ci] = -math.inf if is_minimizing else math.inf
                    #     continue
                    xb = s._mpisppy_model.xbars[ndn_i].value
                    # check variance of xb to determine if consensus achieved
                    var_xb = pyo.value(s._mpisppy_model.xsqbars[ndn_i]) - xb * xb
                    # TODO: How to set this?
                    # TODO: Does this eliminate need for close_to_lb_or_ub? --yes
                    if var_xb  > self.consensus_threshold * self.consensus_threshold:
                        #if self.opt.cylinder_rank == 0 and self.opt.options['verbose']:
                        #print(f"Variance of xbar for {xvar.name} is {var_xb}, consensus not achieved")
                        # if self.opt.cylinder_rank == 0:
                        #     print(f'rc of var {xvar.name}  is {sub.rc[xvar]}')
                        rc[ci] = np.nan
                        continue

                    if is_minimizing:
                        if xb - xvar.lb <= self.bound_tol:
                            rc[ci] += sub._mpisppy_probability * sub.rc[xvar]
                        elif xvar.ub - xb <= self.bound_tol:
                            rc[ci] += sub._mpisppy_probability * sub.rc[xvar]
                        # not close to either bound -> rc = nan
                        else:
                            rc[ci] = np.nan
                    # maximizing
                    else:
                        if xb - xvar.lb <= self.bound_tol:
                            rc[ci] += sub._mpisppy_probability * sub.rc[xvar]
                        elif xvar.ub - xb <= self.bound_tol:
                            rc[ci] += sub._mpisppy_probability * sub.rc[xvar]
                        else:
                            rc[ci] = np.nan

        #print(f"rc: {rc}")
        rcg = np.zeros(self.nonant_length)
        self.cylinder_comm.Allreduce(rc, rcg, op=MPI.SUM)
        self._bound[1:1+self.nonant_length] = rcg
        # if self.opt.cylinder_rank == 0: print(f"in spoke before, rcs: {self._bound[1:1+self.nonant_length]}")