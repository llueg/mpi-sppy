# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import sys
import os
import copy
import netdes

from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension
from mpisppy.extensions.reduced_costs_fixer import ReducedCostsFixer
from mpisppy.extensions.extension import MultiExtension
from mpisppy.extensions.fixer import Fixer

write_solution = True

def _parse_args():
    cfg = config.Config()
    cfg.num_scens_optional()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.xhatlooper_args()
    cfg.ph_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.subgradient_args()
    cfg.xhatshuffle_args()
    cfg.slammax_args()
    cfg.cross_scenario_cuts_args()
    cfg.reduced_costs_args()
    cfg.tracking_args()
    cfg.mip_options()
    cfg.add_to_config("instance_name",
                        description="netdes instance name (e.g., network-10-20-L-01)",
                        domain=str,
                        default=None)                
    cfg.parse_command_line("netdes_cylinders")
    return cfg
    
def main():
    cfg = _parse_args()

    inst = cfg.instance_name
    num_scen = int(inst.split("-")[-3])
    if cfg.num_scens is not None and cfg.num_scens != num_scen:
        raise RuntimeError(f"Argument num-scens={cfg.num_scens} does not match the number "
                           "implied by instance name={num_scen} "
                           "\n(--num-scens is not needed for netdes)")

    fwph = cfg.fwph
    xhatlooper = cfg.xhatlooper
    xhatshuffle = cfg.xhatshuffle
    lagrangian = cfg.lagrangian
    subgradient = cfg.subgradient
    slammax = cfg.slammax
    cross_scenario_cuts = cfg.cross_scenario_cuts
    reduced_costs = cfg.reduced_costs

    if cfg.default_rho is None:
        raise RuntimeError("The --default-rho option must be specified")

    path = f"{netdes.__file__[:-10]}/data/{inst}.dat"
    scenario_creator = netdes.scenario_creator
    scenario_denouement = netdes.scenario_denouement    
    all_scenario_names = [f"Scen{i}" for i in range(num_scen)]
    scenario_creator_kwargs = {"path": path}

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    if cross_scenario_cuts:
        ph_ext = CrossScenarioExtension
    else:
        ph_ext = MultiExtension

    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=ph_ext,
                              rho_setter = None)
    
    ext_classes = []
    if reduced_costs:
        ext_classes.append(ReducedCostsFixer)
    hub_dict["opt_kwargs"]["extension_kwargs"] = {"ext_classes" : ext_classes}

    if cross_scenario_cuts:
        hub_dict["opt_kwargs"]["options"]["cross_scen_options"]\
            = {"check_bound_improve_iterations" : cfg.cross_scenario_iter_cnt}

    
    if reduced_costs:
        hub_dict["opt_kwargs"]["options"]["rc_options"] = {
            "verbose": cfg.rc_verbose,
            "use_rc_fixer": cfg.rc_fixer,
            "zero_rc_tol": cfg.rc_zero_rc_tol,
            "fix_fraction_target_iter0": cfg.rc_fix_fraction_iter0,
            "fix_fraction_target_iterK": cfg.rc_fix_fraction_iterK,
            "progressive_fix_fraction": cfg.rc_progressive_fix_fraction,
            "use_rc_bt": cfg.rc_bound_tightening,
            "bound_tol": cfg.rc_bound_tol,
            "track_rc": cfg.rc_track_rc,
            "track_prefix": cfg.rc_track_prefix
        }
    # FWPH spoke
    if fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = None)

    if subgradient:
        subgradient_spoke = vanilla.subgradient_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = None)
        
    # xhat looper bound spoke
    if xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # slam up bound spoke
    if slammax:
        slammax_spoke = vanilla.slammax_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # cross scenario cuts spoke
    if cross_scenario_cuts:
        cross_scenario_cuts_spoke = vanilla.cross_scenario_cuts_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    if reduced_costs:
        reduced_costs_spoke = vanilla.reduced_costs_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = None)

    list_of_spoke_dict = list()
    if fwph:
        list_of_spoke_dict.append(fw_spoke)
    if lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if subgradient:
        list_of_spoke_dict.append(subgradient_spoke)
    if xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)
    if slammax:
        list_of_spoke_dict.append(slammax_spoke)
    if cross_scenario_cuts:
        list_of_spoke_dict.append(cross_scenario_cuts_spoke)
    if reduced_costs:
        list_of_spoke_dict.append(reduced_costs_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if write_solution:
        wheel.write_first_stage_solution('netdes_build.csv')
        wheel.write_tree_solution('netdes_full_solution')


if __name__ == "__main__":
    main()
