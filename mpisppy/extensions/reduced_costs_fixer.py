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
        self.verbose = False

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
        self.fix_fraction_target = 0.8

        self.bound_tol = 1e-6

        # for updates
        self._last_serial_number = -1
        self._proved_fixed_vars = 0
        self._heuristic_fixed_vars = 0
        self._integer_proved_fixed_nonants = set()

    def pre_iter0(self):
        self._modeler_fixed_nonants = set()
        self._integer_nonants = []
        self.nonant_length = self.opt.nonant_length

        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants.add(ndn_i)
                elif xvar.is_integer():
                    self._integer_nonants.append(ndn_i)
            break
        # print(f"Extension: nonant_length: {self.nonant_length}, integer_nonant_length: {len(self._integer_nonants)}")
    
    def initialize_spoke_indices(self):
        for (i, spoke) in enumerate(self.opt.spcomm.spokes):
            if spoke["spoke_class"] == ReducedCostsSpoke:
                self.reduced_costs_spoke_index = i + 1

    def sync_with_spokes(self):
        spcomm = self.opt.spcomm
        idx = self.reduced_costs_spoke_index
        serial_number = int(round(spcomm.outerbound_receive_buffers[idx][-1]))
        # print(f"serial_number: {serial_number}")
        if serial_number > self._last_serial_number:
            self._last_serial_number = serial_number
            reduced_costs = spcomm.outerbound_receive_buffers[idx][1:1+self.nonant_length]
            integer_cutoffs = spcomm.outerbound_receive_buffers[idx][1+self.nonant_length:-1]
            this_bound = spcomm.outerbound_receive_buffers[idx][0]
            # if self.opt.cylinder_rank == 0: print(f"in extension, rcs: {reduced_costs}")
            # if self.opt.cylinder_rank == 0: print(f"in extension, cutoffs: {integer_cutoffs}")
            self.integer_cutoff_fixing(integer_cutoffs)
            self.reduced_costs_fixing(reduced_costs)
        else:
            if self.opt.cylinder_rank == 0:
                print(f"Total unique vars fixed by reduced cost: {int(round(self._proved_fixed_vars))}")
                print("No new reduced costs!")
                print(f"Total unique vars fixed by heuristic: {int(round(self._heuristic_fixed_vars))}")

    def integer_cutoff_fixing(self, integer_cutoffs):

        raw_fixed_this_iter = 0
        inner_bound = self.opt.spcomm.BestInnerBound
        for sub in self.opt.local_subproblems.values():
            persistent_solver = is_persistent(sub._solver_plugin)
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                # print(f"in scenario: {sn}")
                for ci, ndn_i in enumerate(self._integer_nonants):
                    # TODO: fix for maximization
                    if (sn, ndn_i) in self._integer_proved_fixed_nonants:
                        continue
                    update_var = False
                    if integer_cutoffs[ci] > inner_bound:
                        xb = s._mpisppy_model.xbars[ndn_i].value
                        xvar = s._mpisppy_data.nonant_indices[ndn_i]
                        if (xb - xvar.lb <= self.bound_tol):
                            xvar.fix(xvar.lb)
                            if self.verbose and self.opt.cylinder_rank == 0:
                                print(f"fixing var {xvar.name} to lb {xvar.lb}; cutoff is {integer_cutoffs[ci]} LP-LR")
                            update_var = True
                            raw_fixed_this_iter += 1
                            self._integer_proved_fixed_nonants.add((sn, ndn_i))
                        elif (xvar.ub - xb <= self.bound_tol):
                            xvar.fix(xvar.ub)
                            if self.verbose and self.opt.cylinder_rank == 0:
                                print(f"fixing var {xvar.name} to ub {xvar.ub}; cutoff is {integer_cutoffs[ci]} LP-LR")
                            update_var = True
                            raw_fixed_this_iter += 1
                            self._integer_proved_fixed_nonants.add((sn, ndn_i))
                        else:
                            if self.verbose and self.opt.cylinder_rank == 0:
                                print(f"Could not fix {xvar.name} to bound; cutoff is {integer_cutoffs[ci]} LP-LR, xbar: {xb}")
                    if update_var and persistent_solver:
                        sub._solver_plugin.update_var(xvar)
        self._proved_fixed_vars += raw_fixed_this_iter / len(self.opt.local_scenarios)
        if self.opt.cylinder_rank == 0:
            print(f"Total unique vars fixed by reduced cost: {int(round(self._proved_fixed_vars))}")


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

        if self.opt.cylinder_rank == 0:
            print(f"Heuristic fixing reduced cost cutoff: {target}")

        raw_fixed_this_iter = 0
        inf = float("inf")
        spcomm = self.opt.spcomm
        for sub in self.opt.local_subproblems.values():
            persistent_solver = is_persistent(sub._solver_plugin)
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    if ndn_i in self._modeler_fixed_nonants:
                        continue
                    this_expected_rc = abs_reduced_costs[ci]
                    update_var = False
                    if np.isnan(this_expected_rc):
                        # is nan, variable is not converged in LP-LR
                        if xvar.fixed:
                            xvar.unfix()
                            update_var = True
                            raw_fixed_this_iter -= 1
                            if self.verbose and self.opt.cylinder_rank == 0:
                                print(f"unfixing var {xvar.name}; not converged in LP-LR")
                    else: # not nan, variable is converged in LP-LR
                        if xvar.fixed:
                            if this_expected_rc <= self.zero_rc_tol:
                                xvar.unfix()
                                update_var = True
                                raw_fixed_this_iter -= 1
                                if self.verbose and self.opt.cylinder_rank == 0:
                                    print(f"unfixing var {xvar.name}; reduced cost is zero in LP-LR")
                        else:
                            xb = s._mpisppy_model.xbars[ndn_i].value
                            # TODO: handle maximization case
                            if (this_expected_rc >= target):
                                if (reduced_costs[ci] > 0) and (xb - xvar.lb <= self.bound_tol):
                                    xvar.fix(xvar.lb)
                                    if self.verbose and self.opt.cylinder_rank == 0:
                                        print(f"fixing var {xvar.name} to lb {xvar.lb}; reduced cost is {reduced_costs[ci]} LP-LR")
                                    update_var = True
                                    raw_fixed_this_iter += 1
                                elif (reduced_costs[ci] < 0) and (xvar.ub - xb <= self.bound_tol):
                                    xvar.fix(xvar.ub)
                                    if self.verbose and self.opt.cylinder_rank == 0:
                                        print(f"fixing var {xvar.name} to ub {xvar.ub}; reduced cost is {reduced_costs[ci]} LP-LR")
                                    update_var = True
                                    raw_fixed_this_iter += 1
                    if update_var and persistent_solver:
                        sub._solver_plugin.update_var(xvar)

        self._heuristic_fixed_vars += raw_fixed_this_iter / len(self.opt.local_scenarios)
        if self.opt.cylinder_rank == 0:
            print(f"Total unique vars fixed by heuristic: {int(round(self._heuristic_fixed_vars))}")
