# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import pyomo.environ as pyo
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

        nonant_length = self.opt.nonant_length

        self._make_windows(1 + nonant_length, vbuflen)
        self._locals = np.zeros(vbuflen + 1)
        # over load the _bound attribute here
        # so the rest of the class works as expected
        # first float will be the bound we're sending
        # indices 1:-1 will be the reduced costs, and
        # the last index will be the serial number
        self._bound = np.zeros(1 + nonant_length + 1)

    @property
    def rc(self):
        return self._bound[1:-1]

    @rc.setter
    def rc(self, vals):
        self._bound[1:-1] = vals

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
        self.opt.subproblem_creation(verbose)

        relax_integer_vars = pyo.TransformationFactory("core.relax_integer_vars")
        for s in self.opt.local_subproblems.values():
            relax_integer_vars.apply_to(s)
            s.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        self.opt._create_solvers()

    def lagrangian(self):
        bound = super().lagrangian()
        if bound is not None:
            self.extract_and_store_reduced_costs()
        return bound

    def extract_and_store_reduced_costs(self):
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
                    if xvar.fixed:
                        rc[ci] = np.nan
                        continue
                    xb = s._mpisppy_model.xbars[ndn_i].value
                    if xb - xvar.lb <= self.bound_tol:
                        if rc[ci] < 0: # prior was ub
                            rc[ci] = np.nan
                        else:
                            rc[ci] += sub._mpisppy_probability * pyo.value(sub.rc[xvar])
                    elif xvar.ub - xb <= self.bound_tol:
                        if rc[ci] > 0: # prior was lb
                            rc[ci] = np.nan
                        else:
                            rc[ci] += sub._mpisppy_probability * pyo.value(sub.rc[xvar])

        self.cylinder_comm.Allreduce(rc, self.rc, op=MPI.SUM)

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
