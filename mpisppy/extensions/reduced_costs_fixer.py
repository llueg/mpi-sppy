# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import numpy as np

from mpisppy.extensions.extension import Extension
from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke 

class ReducedCostsFixer(Extension):

    def __init__(self, spobj):
        super().__init__(spobj)
        # TODO: options
    
    def initialize_spoke_indices(self):
        for (i, spoke) in enumerate(self.opt.spcomm.spokes):
            if spoke["spoke_class"] == ReducedCostsSpoke:
                self.reduced_costs_spoke_index = i + 1

    def sync_with_spokes(self):
        spcomm = self.opt.spcomm
        idx = self.reduced_costs_spoke_index
        # TODO: Figure out how not to do extra work if the RCs aren't new
        #       The Hub class doesn't keep track of when information is new 
        self.reduced_costs_fixing(spcomm.outerbound_receive_buffers[idx][1:-1])

    def reduced_costs_fixing(self, reduced_costs):
        print("Expected reduced costs received:")
        ci = 0
        printed = 0
        for sub in self.opt.local_subproblems.values():
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                    this_expected_rc = reduced_costs[ci]
                    if not np.isnan(this_expected_rc):#and abs(this_expected_rc) > 1e-1*abs(self.hub_inner_bound - self.hub_outer_bound):
                        print(f"\t{xvar.name}, rc: {this_expected_rc}, xbar: {s._mpisppy_model.xbars[ndn_i].value}")
                        printed += 1
                    ci += 1
                    if printed > 5:
                        break
