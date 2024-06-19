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

        ph_options = spobj.options
        rc_options = ph_options['rc_options']
        self.verbose = ph_options['verbose'] or rc_options['verbose']

        self._use_rc_bt = rc_options['use_rc_bt']
        # reduced costs less than
        # this in absolute value
        # will be considered 0
        self.zero_rc_tol = rc_options['zero_rc_tol']
        self._use_rc_fixer = rc_options['use_rc_fixer']
        # Percentage of variables which are at the bound we will target
        # to fix. We never fix varibles with reduced costs less than
        # the `zero_rc_tol` in absolute value
        self._fix_fraction_target_iter0 = rc_options['fix_fraction_target_iter0']
        self._fix_fraction_target_iterK = rc_options['fix_fraction_target_iterK']
        self._progressive_fix_fraction = rc_options['progressive_fix_fraction']
        self.fix_fraction_target = self._fix_fraction_target_iter0

        # TODO: This should be same as in rc spoke?
        self.bound_tol = rc_options['bound_tol']

        # TODO: move tracking to phtracker?
        self._track_rc = True
        self._track_rc = (self.opt.cylinder_rank == 0) and self._track_rc
        # dict with iter
        self._rc_history = {}
        self._rc_history = []
        self._target_history = []

        if not (self._use_rc_bt or self._use_rc_fixer) and \
            self.opt.cylinder_rank == 0:
            print(f"Warning: ReducedCostsFixer will be idle. Enable use_rc_bt or use_rc_fixer in options.")

        self._options = rc_options

        # for updates
        self._last_serial_number = -1
        #self._proved_fixed_vars = 0
        self._heuristic_fixed_vars = 0
        #self._integer_proved_fixed_nonants = set()

    def pre_iter0(self):
        self._modeler_fixed_nonants = set()
        self._integer_nonants = []
        self.nonant_length = self.opt.nonant_length
        var_names = []
        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants.add(ndn_i)
                    continue
                if xvar.is_integer():
                    self._integer_nonants.append(ndn_i)
                var_names.append((k, xvar.name))
            break
    
        self.var_names = var_names

    def post_iter0(self):
        self.fix_fraction_target = self._fix_fraction_target_iterK

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
            this_outer_bound = spcomm.outerbound_receive_buffers[idx][0]
            # if self.opt.cylinder_rank == 0: print(f"in extension, rcs: {reduced_costs}")
            if self._track_rc:
                self._rc_history.append(reduced_costs)
            
            self.reduced_costs_bounds_tightening(reduced_costs, this_outer_bound)
            if self._use_rc_fixer:
                self.reduced_costs_fixing(reduced_costs)
            
            if self.opt.cylinder_rank == 0 and self._track_rc:
                rc_array = np.array(self._rc_history)
                #np.save(f"rc_tracking_test/rc_history.npy", rc_array)
                target_array = np.array(self._target_history)
                #np.save(f"rc_tracking_test/target_history.npy", target_array)

        else:
            if self.opt.cylinder_rank == 0:
                print("No new reduced costs!")


    def reduced_costs_bounds_tightening(self, reduced_costs, this_outer_bound):

        bounds_reduced_this_iter = 0
        inner_bound = self.opt.spcomm.BestInnerBound
        outer_bound = this_outer_bound
        is_minimizing = self.opt.is_minimizing
        if np.isinf(inner_bound) or np.isinf(outer_bound):
            if self.opt.cylinder_rank == 0:
                print(f"Bounds tightened by reduced cost: 0 (inner or outer bound not available)")
            return
        # TODO: keep track of and skip already fixed variables, i.e. lb=ub
        for sub in self.opt.local_subproblems.values():
            persistent_solver = is_persistent(sub._solver_plugin)
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    if ndn_i in self._modeler_fixed_nonants:
                        continue
                    this_expected_rc = reduced_costs[ci]
                    update_var = False
                    if np.isnan(this_expected_rc) or np.isinf(this_expected_rc):
                        continue

                    # TODO: could simplify if/else blocks using sign variable - might reduce readability?
                    # alternatively, could move some blocks into functions
                    if is_minimizing:
                        # var at lb
                        if this_expected_rc > 0 + self.zero_rc_tol:
                            new_ub = xvar.lb + (inner_bound - outer_bound)/ this_expected_rc
                            old_ub = xvar.ub
                            if new_ub < old_ub:
                                if ndn_i in self._integer_nonants:
                                    new_ub = np.floor(new_ub)
                                    xvar.setub(new_ub)
                                else:
                                    xvar.setub(new_ub)
                                if self.verbose and self.opt.cylinder_rank == 0:
                                    print(f"tightening ub of var {xvar.name} to {new_ub} from {old_ub}; reduced cost is {this_expected_rc}")
                                update_var = True
                                bounds_reduced_this_iter += 1
                        # var at ub
                        elif this_expected_rc < 0 - self.zero_rc_tol:
                            new_lb = xvar.ub + (inner_bound - outer_bound)/ this_expected_rc
                            old_lb = xvar.lb
                            if new_lb > old_lb:
                                if ndn_i in self._integer_nonants:
                                    new_lb = np.ceil(new_lb)
                                    xvar.setlb(new_lb)
                                else:
                                    xvar.setlb(new_lb)
                                if self.verbose and self.opt.cylinder_rank == 0:
                                    print(f"tightening lb of var {xvar.name} to {new_lb} from {old_lb}; reduced cost is {this_expected_rc}")
                                update_var = True
                                bounds_reduced_this_iter += 1
                    # maximization
                    else:
                        # var at lb
                        if this_expected_rc < 0 - self.zero_rc_tol:
                            new_ub = xvar.lb - (outer_bound - inner_bound)/ this_expected_rc
                            old_ub = xvar.ub
                            if new_ub < old_ub:
                                if ndn_i in self._integer_nonants:
                                    new_ub = np.floor(new_ub)
                                    xvar.setub(new_ub)
                                else:
                                    xvar.setub(new_ub)
                                if self.verbose and self.opt.cylinder_rank == 0:
                                    print(f"tightening ub of var {xvar.name} to {new_ub} from {old_ub}; reduced cost is {this_expected_rc}")
                                update_var = True
                                bounds_reduced_this_iter += 1
                        # var at ub
                        elif this_expected_rc > 0 + self.zero_rc_tol:
                            new_lb = xvar.ub - (outer_bound - inner_bound)/ this_expected_rc
                            old_lb = xvar.lb
                            if new_lb > old_lb:
                                if ndn_i in self._integer_nonants:
                                    new_lb = np.ceil(new_lb)
                                    xvar.setlb(new_lb)
                                else:
                                    xvar.setlb(new_lb)
                                if self.verbose and self.opt.cylinder_rank == 0:
                                    print(f"tightening lb of var {xvar.name} to {new_lb} from {old_lb}; reduced cost is {this_expected_rc}")
                                update_var = True
                                bounds_reduced_this_iter += 1

                    if update_var and persistent_solver:
                        sub._solver_plugin.update_var(xvar)

        total_bounds_tightened = bounds_reduced_this_iter / len(self.opt.local_scenarios)
        if self.opt.cylinder_rank == 0:
            print(f"Bounds tightened by reduced cost: {int(round(total_bounds_tightened))}/{self.nonant_length}")


    def reduced_costs_fixing(self, reduced_costs):

        if np.all(np.isnan(reduced_costs)):
            print("All reduced costs are nan, heuristic fixing will not be applied")
            return
        
        # compute the quantile target
        abs_reduced_costs = np.abs(reduced_costs)

        if self._progressive_fix_fraction:
            # TODO: relies on heuristic_fixed_vars to be accurate
            already_fixed_frac = np.minimum(self._heuristic_fixed_vars / self.nonant_length, 1)
            additional_fix_frac = (1 - already_fixed_frac) * self.fix_fraction_target
            fix_fraction_target = already_fixed_frac + additional_fix_frac
        else:
            fix_fraction_target = self.fix_fraction_target

        if fix_fraction_target > 0:
            target = np.nanquantile(abs_reduced_costs, 1 - fix_fraction_target, method="median_unbiased")
        else:
            target = float("inf")
        if target < self.zero_rc_tol:
            target = self.zero_rc_tol

        if self._track_rc:
            self._target_history.append(target)

        if self.opt.cylinder_rank == 0:
            print(f"Heuristic fixing reduced cost cutoff: {target}")

        raw_fixed_this_iter = 0
        for sub in self.opt.local_subproblems.values():
            # TODO: does this still work without the persistent solver?
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
                            if (this_expected_rc <= target):
                                xvar.unfix()
                                update_var = True
                                raw_fixed_this_iter -= 1
                                if self.verbose and self.opt.cylinder_rank == 0:
                                    print(f"unfixing var {xvar.name}; reduced cost is zero/below target in LP-LR")
                        else:
                            xb = s._mpisppy_model.xbars[ndn_i].value
                            if (this_expected_rc >= target):
                                if self.opt.is_minimizing:
                                    # TODO: First check can be simplified as abs(rc) is already checked above
                                    if (reduced_costs[ci] > 0 + self.zero_rc_tol) and (xb - xvar.lb <= self.bound_tol):
                                        xvar.fix(xvar.lb)
                                        if self.verbose and self.opt.cylinder_rank == 0:
                                            print(f"fixing var {xvar.name} to lb {xvar.lb}; reduced cost is {reduced_costs[ci]} LP-LR")
                                        update_var = True
                                        raw_fixed_this_iter += 1
                                    elif (reduced_costs[ci] < 0 - self.zero_rc_tol) and (xvar.ub - xb <= self.bound_tol):
                                        xvar.fix(xvar.ub)
                                        if self.verbose and self.opt.cylinder_rank == 0:
                                            print(f"fixing var {xvar.name} to ub {xvar.ub}; reduced cost is {reduced_costs[ci]} LP-LR")
                                        update_var = True
                                        raw_fixed_this_iter += 1
                                    else:
                                        # rc is near 0 or 
                                        # xbar from MIP might differ from rc from relaxation
                                        #print(f'unexpected rc value: scen {sn}, var {xvar.name}, rc = {reduced_costs[ci]}, {xvar.lb} <= {xb:.7f} <= {xvar.ub}')
                                        pass
                                else:
                                    if (reduced_costs[ci] < 0 - self.zero_rc_tol) and (xb - xvar.lb <= self.bound_tol):
                                        xvar.fix(xvar.lb)
                                        if self.verbose and self.opt.cylinder_rank == 0:
                                            print(f"fixing var {xvar.name} to lb {xvar.lb}; reduced cost is {reduced_costs[ci]} LP-LR")
                                        update_var = True
                                        raw_fixed_this_iter += 1
                                    elif (reduced_costs[ci] > 0 + self.zero_rc_tol) and (xvar.ub - xb <= self.bound_tol):
                                        xvar.fix(xvar.ub)
                                        if self.verbose and self.opt.cylinder_rank == 0:
                                            print(f"fixing var {xvar.name} to ub {xvar.ub}; reduced cost is {reduced_costs[ci]} LP-LR")
                                        update_var = True
                                        raw_fixed_this_iter += 1
                                    else:
                                        # rc is near 0 or 
                                        # xbar from MIP might differ from rc from relaxation
                                        #print(f'unexpected rc value: scen {sn}, var {xvar.name}, rc = {reduced_costs[ci]}, {xvar.lb} <= {xb:.7f} <= {xvar.ub}')
                                        pass
                    
                    if update_var and persistent_solver:
                        sub._solver_plugin.update_var(xvar)

        self._heuristic_fixed_vars += raw_fixed_this_iter / len(self.opt.local_scenarios)
        if self.opt.cylinder_rank == 0:
            print(f"Total unique vars fixed by heuristic: {int(round(self._heuristic_fixed_vars))}/{self.nonant_length}")
