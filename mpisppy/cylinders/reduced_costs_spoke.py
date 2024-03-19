# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import math
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
import numpy as np
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.utils.sputils import is_persistent 
from mpisppy import MPI

class ReducedCostsSpoke(LagrangianOuterBound):

    converger_spoke_char = 'R'
    bound_tol = 1e-6

    def make_windows(self):
        if not hasattr(self.opt, "local_scenarios"):
            raise RuntimeError("Provided SPBase object does not have local_scenarios attribute")

        if len(self.opt.local_scenarios) == 0:
            raise RuntimeError("Rank has zero local_scenarios")

        vbuflen = 2
        for s in self.opt.local_scenarios.values():
            vbuflen += len(s._mpisppy_data.nonant_indices)

        self.nonant_length = self.opt.nonant_length

        # collect the vars original integer for later, and count how many
        if self.opt.is_minimizing:
            default_best_incumbent = -math.inf
            self._update_best = _update_best_cutoff_minimizing
        else:
            default_best_incumbent = math.inf
            self._update_best = _update_best_cutoff_maximizing

        self._modeler_fixed_nonants = set()
        self._integer_best_incumbent_to_fix = {}
        self._integer_proved_fixed_nonants = set()

        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants.add(ndn_i)
                elif xvar.is_integer():
                    self._integer_best_incumbent_to_fix[ndn_i] = default_best_incumbent
            break
        #print(f"self._modeler_fixed_nonants: {self._modeler_fixed_nonants}")
        #print(f"self._integer_best_incumbent_to_fix: {self._integer_best_incumbent_to_fix}")

        self.integer_nonant_length = len(self._integer_best_incumbent_to_fix)

        self._make_windows(1 + self.nonant_length + self.integer_nonant_length, vbuflen)
        self._locals = np.zeros(vbuflen + 1)
        # set the initial local inner / outer bounds to a valid value
        if self.opt.is_minimizing:
            self._locals[-2] = math.inf
            self._locals[-3] = -math.inf
        else:
            self._locals[-2] = -math.inf
            self._locals[-3] = math.inf
        # over load the _bound attribute here
        # so the rest of the class works as expected
        # first float will be the bound we're sending
        # indices 1:-1 will be the reduced costs, and
        # the last index will be the serial number
        self._bound = np.zeros(1 + self.nonant_length + self.integer_nonant_length + 1)
        print(f"nonant_length: {self.nonant_length}, integer_nonant_length: {self.integer_nonant_length}")

    @property
    def rc(self):
        return self._bound[1:1+self.nonant_length]

    @rc.setter
    def rc(self, vals):
        self._bound[1:1+self.nonant_length] = vals

    @property
    def integer_cutoff(self):
        return self._bound[1+self.nonant_length:-1]

    @rc.setter
    def integer_cutoff(self, vals):
        self._bound[1+self.nonant_length:-1] = vals

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


    def update_integer_var_cache(self, this_bound, reduced_costs):
        for k,s in self.opt.local_scenarios.items():
            for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                if ndn_i not in self._integer_best_incumbent_to_fix:
                    continue
                if ndn_i in self._integer_proved_fixed_nonants:
                    continue
                this_expected_rc = reduced_costs[ci]
                if np.isnan(this_expected_rc):
                    continue
                self._integer_best_incumbent_to_fix[ndn_i] = self._update_best(
                    self._integer_best_incumbent_to_fix[ndn_i],
                    this_bound,
                    this_expected_rc,
                )
                #print(f"{xvar.name}, cutoff: {self._integer_best_incumbent_to_fix[xvar]}")
            break

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
        rc = np.zeros(len(self.rc))
        for sub in self.opt.local_subproblems.values():
            if is_persistent(sub._solver_plugin):
                # TODO: only load nonant's RC
                # TODO: what happens with non-persistent solvers?
                sub._solver_plugin.load_rc()
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    #print(f"{xvar.name}, rc: {pyo.value(sub.rc[xvar])}, val: {xvar.value}, lb: {xvar.lb}, ub: {xvar.ub} ")
                    # fixed by modeler
                    if ndn_i in self._modeler_fixed_nonants:
                        rc[ci] = np.nan
                        continue
                    if ndn_i in self._integer_proved_fixed_nonants:
                        # TODO: needs to be fixed for maximization problems
                        if xvar.value == xvar.lb:
                            rc[ci] = math.inf
                        else:
                            rc[ci] = -math.inf
                        continue
                    xb = s._mpisppy_model.xbars[ndn_i].value
                    # TODO: needs to be fixed for maximization problems
                    if xb - xvar.lb <= self.bound_tol:
                        if rc[ci] < 0: # prior was ub
                            rc[ci] = np.nan
                        else:
                            rc[ci] += sub._mpisppy_probability * sub.rc[xvar]
                    elif xvar.ub - xb <= self.bound_tol:
                        if rc[ci] > 0: # prior was lb
                            rc[ci] = np.nan
                        else:
                            rc[ci] += sub._mpisppy_probability * sub.rc[xvar]

        self.cylinder_comm.Allreduce(rc, self.rc, op=MPI.SUM)

        if len(self._integer_best_incumbent_to_fix) == 0:
            return

        inner_bound = self.hub_inner_bound
        print(f"spoke inner_bound: {inner_bound}")
        print(f"this outer_bound: {outer_bound}")

        # now try to prove things based on the best inner bound
        self.update_integer_var_cache(outer_bound, rc)
        for sub in self.opt.local_subproblems.values():
            persistent_solver = is_persistent(sub._solver_plugin)
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                for ci, (ndn_i, val) in enumerate(self._integer_best_incumbent_to_fix.items()):
                    self.integer_cutoff[ci] = val
                    #if val > outer_bound*(1+1e-4): 
                    #    print(f"var {s._mpisppy_data.nonant_indices[ndn_i].name}, cutoff is {val}")
                    #if val > inner_bound and ndn_i not in self._integer_proved_fixed_nonants:
                    #    self._integer_proved_fixed_nonants.add(ndn_i)
                    #    xvar = s._mpisppy_data.nonant_indices[ndn_i]
                    #    if (xb - xvar.lb) <= self.bound_tol:
                    #        xvar.fix(xvar.lb)
                    #    else:
                    #        assert (xvar.ub - xb) <= self.bound_tol
                    #        xvar.fix(xvar.ub)
                    #    sub._solver_plugin.update_var(xvar)

        print(f"in spoke, rcs: {self.rc}")
        print(f"in spoke, cutoffs: {self.integer_cutoff}")


        #if self.cylinder_rank == 0:
        #    print("Expected reduced costs sent:")
        #    for sub in self.opt.local_subproblems.values():
        #        for sn in sub.scen_list:
        #            s = self.opt.local_scenarios[sn]
        #            for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
        #                this_expected_rc = self.rc[ci]
        #                if not np.isnan(this_expected_rc):#and abs(this_expected_rc) > 1e-1*abs(self.hub_inner_bound - self.hub_outer_bound):
        #                    print(f"\t{xvar.name}, rc: {this_expected_rc}, xbar: {s._mpisppy_model.xbars[ndn_i].value}")
        #            break


def _update_best_cutoff_minimizing(current_best, best_bound, rc):
    if rc < 0: # at ub, so decreasing the var
        new_best = best_bound - rc
    else: # at lb, so increasing the var
        new_best = best_bound + rc
    return max(current_best, new_best)


def _update_best_cutoff_maximizing(current_best, best_bound, rc):
    if rc > 0: # at ub, so decreasing the var
        new_best = best_bound - rc
    else: # at lb, so increasing the var
        new_best = best_bound + rc
    return min(current_best, new_best)
