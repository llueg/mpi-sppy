# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import numpy as np

from pyomo.common.collections import ComponentSet, ComponentMap

from mpisppy.extensions.extension import Extension
from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke 
from mpisppy.utils.sputils import is_persistent

class ReducedCostsFixer(Extension):

    def __init__(self, spobj):
        super().__init__(spobj)
        # TODO: expose options

        # reduced costs less than
        # this in absolute value
        # will be considered 0
        self.zero_rc_tol = 1e-6

        # Percentage of variables which
        # are at the bound we will target
        # to fix. We never fix varibles
        # with reduced costs less than
        # the `zero_rc_tol` in absolute
        # value

        # TODO: may want a different one
        #       for iteration 0
        # TODO: seems very, very likely
        self.fix_fraction_target = 0.0

        self.bound_tol = 1e-6

        # for updates
        self._last_serial_number = -1
        self._total_fixed_vars = 0
        self._modeler_fixed_vars = ComponentSet()
        self._integer_best_incumbent_to_fix = ComponentMap()

    def pre_iter0(self):
        if self.opt.is_minimizing:
            default_best_incumbent = float("-inf")
            self._update_best = _update_best_cutoff_minimizing
        else:
            default_best_incumbent = float("inf")
            self._update_best = _update_best_cutoff_maximizing
        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_vars.add(xvar)
                elif xvar.is_integer():
                    self._integer_best_incumbent_to_fix[xvar] = default_best_incumbent
    
    def initialize_spoke_indices(self):
        for (i, spoke) in enumerate(self.opt.spcomm.spokes):
            if spoke["spoke_class"] == ReducedCostsSpoke:
                self.reduced_costs_spoke_index = i + 1

    def sync_with_spokes(self):
        spcomm = self.opt.spcomm
        idx = self.reduced_costs_spoke_index
        serial_number = int(round(spcomm.outerbound_receive_buffers[idx][-1]))
        print(f"serial_number: {serial_number}")
        if serial_number > self._last_serial_number:
            self._last_serial_number = serial_number
            reduced_costs = spcomm.outerbound_receive_buffers[idx][1:-1]
            this_bound = spcomm.outerbound_receive_buffers[idx][0]
            self.update_integer_var_cache(this_bound, reduced_costs)
            self.reduced_costs_fixing(reduced_costs)

    def update_integer_var_cache(self, this_bound, reduced_costs):
        for k,s in self.opt.local_scenarios.items():
            for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                if xvar not in self._integer_best_incumbent_to_fix:
                    continue
                this_expected_rc = reduced_costs[ci]
                if np.isnan(this_expected_rc):
                    continue
                self._integer_best_incumbent_to_fix[xvar] = self._update_best(
                    self._integer_best_incumbent_to_fix[xvar],
                    this_bound,
                    this_expected_rc,
                )
                #print(f"{xvar.name}, cutoff: {self._integer_best_incumbent_to_fix[xvar]}")

    def reduced_costs_fixing(self, reduced_costs):

        # compute the quantile target
        abs_reduced_costs = np.abs(reduced_costs)

        # TODO: maybe the can be adaptive and ignore
        #       presently fixed variables, such that
        #       it becomes more agressive with the same
        #       fixed fraction as the iterations continue
        if self.fix_fraction_target > 0:
            target = np.nanquantile(abs_reduced_costs, 1 - self.fix_fraction_target, method="median_unbiased")
        else:
            target = float("inf")
        if target < self.zero_rc_tol:
            target = self.zero_rc_tol

        print(f"target rc: {target}")

        raw_fixed_this_iter = 0
        inf = float("inf")
        spcomm = self.opt.spcomm
        for sub in self.opt.local_subproblems.values():
            persistent_solver = is_persistent(sub._solver_plugin)
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                print(f"in scenario: {sn}")
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    if xvar in self._modeler_fixed_vars:
                        continue
                    this_expected_rc = abs_reduced_costs[ci]
                    update_var = False
                    if np.isnan(this_expected_rc):
                        # is nan, variable is not converged in LP-LR
                        if xvar.fixed:
                            xvar.unfix()
                            update_var = True
                            raw_fixed_this_iter -= 1
                            print(f"unfixing var {xvar.name}; not converged in LP-LR")
                    else: # not nan, variable is converged in LP-LR
                        if xvar.fixed:
                            if this_expected_rc <= self.zero_rc_tol:
                                xvar.unfix()
                                update_var = True
                                raw_fixed_this_iter -= 1
                                print(f"unfixing var {xvar.name}; reduced cost is zero in LP-LR")
                        else:
                            xb = s._mpisppy_model.xbars[ndn_i].value
                            # TODO: handle maximization case
                            if (this_expected_rc > target) or (self._integer_best_incumbent_to_fix.get(xvar, -inf) > spcomm.BestInnerBound):
                                if (reduced_costs[ci] > 0) and (xb - xvar.lb <= self.bound_tol):
                                    xvar.fix(xvar.lb)
                                    print(f"fixing var {xvar.name} to lb {xvar.lb}; reduced cost is {reduced_costs[ci]} LP-LR")
                                    if xvar in self._integer_best_incumbent_to_fix:
                                        print(f"\tcutoff objective value: {self._integer_best_incumbent_to_fix[xvar]}")
                                    update_var = True
                                    raw_fixed_this_iter += 1
                                elif (reduced_costs[ci] < 0) and (xvar.ub - xb <= self.bound_tol):
                                    xvar.fix(xvar.ub)
                                    print(f"fixing var {xvar.name} to ub {xvar.ub}; reduced cost is {reduced_costs[ci]} LP-LR")
                                    if xvar in self._integer_best_incumbent_to_fix:
                                        print(f"\tcutoff objective value: {self._integer_best_incumbent_to_fix[xvar]}")
                                    update_var = True
                                    raw_fixed_this_iter += 1
                    if update_var and persistent_solver:
                        sub._solver_plugin.update_var(xvar)

        self._total_fixed_vars += raw_fixed_this_iter / len(self.opt.local_scenarios)
        if self.opt.cylinder_rank == 0:
            print(f"Unique vars fixed so far - {self._total_fixed_vars}")


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
