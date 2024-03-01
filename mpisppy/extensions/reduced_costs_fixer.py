# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import numpy as np

from pyomo.common.collections import ComponentSet

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
        self.fix_fraction_target = 0.1

        self.bound_tol = 1e-6

        # for updates
        self._last_serial_number = -1
        self._total_fixed_vars = 0
        self._modeler_fixed_vars = ComponentSet()

    def pre_iter0(self):
        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_vars.add(xvar)
    
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
            self.reduced_costs_fixing(spcomm.outerbound_receive_buffers[idx][1:-1])

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

        abs_gap, _ = self.opt.spcomm.compute_gaps()

        raw_fixed_this_iter = 0
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
                            if (this_expected_rc > target) or (this_expected_rc > abs_gap and xvar.is_integer()):
                                if (reduced_costs[ci] > 0) and (xb - xvar.lb <= self.bound_tol):
                                    xvar.fix(xvar.lb)
                                    print(f"fixing var {xvar.name} to lb {xvar.lb}; reduced cost is {reduced_costs[ci]} LP-LR")
                                    update_var = True
                                    raw_fixed_this_iter += 1
                                elif (reduced_costs[ci] < 0) and (xvar.ub - xb <= self.bound_tol):
                                    xvar.fix(xvar.ub)
                                    print(f"fixing var {xvar.name} to ub {xvar.ub}; reduced cost is {reduced_costs[ci]} LP-LR")
                                    update_var = True
                                    raw_fixed_this_iter += 1
                    if update_var and persistent_solver:
                        sub._solver_plugin.update_var(xvar)

        self._total_fixed_vars += raw_fixed_this_iter / len(self.opt.local_scenarios)
        if self.opt.cylinder_rank == 0:
            print(f"Unique vars fixed so far - {self._total_fixed_vars}")
