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

    converger_spoke_char = 'R'
    # TODO: set option
    bound_tol = 1e-6
    consensus_threshold = 1e-3

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
        # if self.opt.is_minimizing:
        #     default_best_incumbent = -math.inf
        #     self._update_best = _update_best_cutoff_minimizing
        # else:
        #     default_best_incumbent = math.inf
        #     self._update_best = _update_best_cutoff_maximizing

        self._modeler_fixed_nonants = set()
        #self._integer_best_incumbent_to_fix = {}
        # TODO: This is not populated anywhere
        self._integer_proved_fixed_nonants = set()

        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants.add(ndn_i)
                # elif xvar.is_integer():
                #     self._integer_best_incumbent_to_fix[ndn_i] = default_best_incumbent
            break
        #print(f"self._modeler_fixed_nonants: {self._modeler_fixed_nonants}")
        #print(f"self._integer_best_incumbent_to_fix: {self._integer_best_incumbent_to_fix}")

        #self.integer_nonant_length = len(self._integer_best_incumbent_to_fix)

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

    # @property
    # def integer_cutoff(self):
    #     return self._bound[1+self.nonant_length:-1]

    # @rc.setter
    # def integer_cutoff(self, vals):
    #     self._bound[1+self.nonant_length:-1] = vals

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


    # def update_integer_var_cache(self, this_bound, reduced_costs):
    #     for k,s in self.opt.local_scenarios.items():
    #         for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
    #             if ndn_i not in self._integer_best_incumbent_to_fix:
    #                 continue
    #             if ndn_i in self._integer_proved_fixed_nonants:
    #                 continue
    #             this_expected_rc = reduced_costs[ci]
    #             if np.isnan(this_expected_rc):
    #                 continue
    #             self._integer_best_incumbent_to_fix[ndn_i] = self._update_best(
    #                 self._integer_best_incumbent_to_fix[ndn_i],
    #                 this_bound,
    #                 this_expected_rc,
    #             )
    #             # print(f"{xvar.name}, cutoff: {self._integer_best_incumbent_to_fix[xvar]}")
    #         break

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
        # -1: close to lb, 1: close to ub, 0: not encoutered
        close_to_lb_or_ub = np.zeros(self.nonant_length)
        num_total_scenarios = sum(len(sub.scen_list) for sub in self.opt.local_subproblems.values())

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
                    if ndn_i in self._integer_proved_fixed_nonants:
                        if xvar.value == xvar.lb:
                            rc[ci] = math.inf if is_minimizing else -math.inf
                        else:
                            rc[ci] = -math.inf if is_minimizing else math.inf
                        continue
                    xb = s._mpisppy_model.xbars[ndn_i].value
                    # check variance of xb to determine if consensus achieved
                    var_xb = pyo.value(s._mpisppy_model.xsqbars[ndn_i]) - xb * xb
                    # TODO: How to set this?
                    # TODO: Does this eliminate need for close_to_lb_or_ub? --yes
                    # if var_xb  > self.consensus_threshold * self.consensus_threshold:
                    #     if self.opt.cylinder_rank == 0 and self.opt.options['verbose']:
                    #         print(f"Variance of xbar for {xvar.name} is {var_xb}, consensus not achieved")
                    #     rc[ci] = np.nan
                    #     continue

                    if is_minimizing:
                        if xb - xvar.lb <= self.bound_tol:
                            # check if var at different bound before
                            if close_to_lb_or_ub[ci] > 0 or np.isnan(rc[ci]):
                                rc[ci] = np.nan
                                close_to_lb_or_ub[ci] = np.nan
                            else:
                                close_to_lb_or_ub[ci] -= 1
                                # We use prob of subproblem to get appropriate rc for overall solution
                                rc[ci] += sub._mpisppy_probability * sub.rc[xvar]
                        elif xvar.ub - xb <= self.bound_tol:
                            if close_to_lb_or_ub[ci] < 0 or np.isnan(rc[ci]):
                                rc[ci] = np.nan
                                close_to_lb_or_ub[ci] = np.nan
                            else:
                                close_to_lb_or_ub[ci] += 1
                                rc[ci] += sub._mpisppy_probability * sub.rc[xvar]
                        # not close to either bound -> rc = nan?
                        else:
                            rc[ci] = np.nan
                            close_to_lb_or_ub[ci] = np.nan
                    # maximizing
                    else:
                        if xb - xvar.lb <= self.bound_tol:
                            if close_to_lb_or_ub[ci] > 0 or np.isnan(rc[ci]):
                                rc[ci] = np.nan
                                close_to_lb_or_ub[ci] = np.nan
                            else:
                                close_to_lb_or_ub[ci] -= 1
                                rc[ci] += sub._mpisppy_probability * sub.rc[xvar]
                        elif xvar.ub - xb <= self.bound_tol:
                            if close_to_lb_or_ub[ci] < 0 or np.isnan(rc[ci]):
                                rc[ci] = np.nan
                                close_to_lb_or_ub[ci] = np.nan
                            else:
                                close_to_lb_or_ub[ci] -= 1
                                rc[ci] += sub._mpisppy_probability * sub.rc[xvar]
                        else:
                            rc[ci] = np.nan
                            close_to_lb_or_ub[ci] = np.nan

        #print(f"rc: {rc}")
        g_lb_or_ub = np.zeros(self.nonant_length)
        self.cylinder_comm.Allreduce(close_to_lb_or_ub, g_lb_or_ub, op=MPI.SUM)
        inconsistent_bounds = np.asarray(np.abs(g_lb_or_ub) != num_total_scenarios).nonzero()
        rcg = np.zeros(self.nonant_length)
        self.cylinder_comm.Allreduce(rc, rcg, op=MPI.SUM)
        rcg[inconsistent_bounds] = np.nan
        self._bound[1:1+self.nonant_length] = rcg
        # if self.opt.cylinder_rank == 0: print(f"in spoke before, rcs: {self._bound[1:1+self.nonant_length]}")

        # if len(self._integer_best_incumbent_to_fix) == 0:
        #     return

        #inner_bound = self.hub_inner_bound
        #outer_bound = self.hub_outer_bound
        # if self.opt.cylinder_rank == 0: print(f"spoke inner_bound: {inner_bound}")
        # if self.opt.cylinder_rank == 0: print(f"this outer_bound: {outer_bound}")

        # now try to prove things based on the best inner bound
        # self.update_integer_var_cache(outer_bound, rc)
        # integer_cutoff = np.zeros(self.integer_nonant_length)
        # for sub in self.opt.local_subproblems.values():
        #     persistent_solver = is_persistent(sub._solver_plugin)
        #     for sn in sub.scen_list:
        #         s = self.opt.local_scenarios[sn]
        #         for ci, (ndn_i, val) in enumerate(self._integer_best_incumbent_to_fix.items()):
        #             integer_cutoff[ci] = val
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

        # if self.opt.cylinder_rank == 0: print(f"in spoke, cutoffs: {integer_cutoff}")
        # self._bound[1+self.nonant_length:-1] = integer_cutoff
        # if self.opt.cylinder_rank == 0: print(f"in spoke, rcs: {self.rc}")
        # if self.opt.cylinder_rank == 0: print(f"in spoke, cutoffs: {self._bound[1+self.nonant_length:-1]}")


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


# def _update_best_cutoff_minimizing(current_best, best_bound, rc):
#     if rc < 0: # at ub, so decreasing the var
#         new_best = best_bound - rc
#     else: # at lb, so increasing the var
#         new_best = best_bound + rc
#     return max(current_best, new_best) # max not needed as always new_best >= current_best ?


# def _update_best_cutoff_maximizing(current_best, best_bound, rc):
#     if rc > 0: # at ub, so decreasing the var
#         new_best = best_bound - rc
#     else: # at lb, so increasing the var
#         new_best = best_bound + rc
#     return min(current_best, new_best)
