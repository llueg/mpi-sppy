import pyomo.environ as pyo

import mpisppy.phbase


class L1PHBase(mpisppy.phbase.PHBase):

    def __init__(self,
                 options,
                 all_scenario_names,
                 scenario_creator,
                 scenario_denouement=None,
                 all_nodenames=None,
                 mpicomm=None,
                 scenario_creator_kwargs=None, 
                 extensions=None,
                 extension_kwargs=None,
                 ph_converger=None,
                 rho_setter=None,
                 variable_probability=None):
        super().__init__(options,
                         all_scenario_names,
                         scenario_creator,
                         scenario_denouement, 
                         all_nodenames,
                         mpicomm,
                         scenario_creator_kwargs,
                         extensions,
                         extension_kwargs,
                         ph_converger,
                         rho_setter,
                         variable_probability)
        
        self._prox_approx = False



    def attach_PH_to_objective(self, add_duals, add_prox):
        """ Attach dual weight and prox terms to the objective function of the
        models in `local_scenarios`.

        Args:
            add_duals (boolean):
                If True, adds dual weight (Ws) to the objective.
            add_prox (boolean):
                If True, adds the prox term to the objective.
        """

        # if ('linearize_binary_proximal_terms' in self.options):
        #     lin_bin_prox = self.options['linearize_binary_proximal_terms']
        # else:
        #     lin_bin_prox = False
        lin_bin_prox = False
        # if ('linearize_proximal_terms' in self.options):
        #     self._prox_approx = self.options['linearize_proximal_terms']
        #     if 'proximal_linearization_tolerance' in self.options:
        #         self.prox_approx_tol = self.options['proximal_linearization_tolerance']
        #     else:
        #         self.prox_approx_tol = 1.e-1
        # else:
        #     self._prox_approx = False
        self._prox_approx = False

        for (sname, scenario) in self.local_scenarios.items():
            """Attach the dual and prox terms to the objective.
            """
            if ((not add_duals) and (not add_prox)):
                return
            objfct = self.saved_objectives[sname]
            is_min_problem = objfct.is_minimizing()

            xbars = scenario._mpisppy_model.xbars

            if self._prox_approx:
                # set-up pyomo IndexVar, but keep it sparse
                # since some nonants might be binary
                # Define the first cut to be _xsqvar >= 0
                scenario._mpisppy_model.xsqvar = pyo.Var(scenario._mpisppy_data.nonant_indices, dense=False, bounds=(0, None))
                scenario._mpisppy_model.xsqvar_cuts = pyo.Constraint(scenario._mpisppy_data.nonant_indices, pyo.Integers)
                scenario._mpisppy_data.xsqvar_prox_approx = {}
            else:
                scenario._mpisppy_model.xsqvar = None
                scenario._mpisppy_data.xsqvar_prox_approx = False


            # add L1 proximal term
            scenario._mpisppy_model.xdiff_pos = pyo.Var(scenario._mpisppy_data.nonant_indices.keys(), domain=pyo.NonNegativeReals)
            scenario._mpisppy_model.xdiff_neg = pyo.Var(scenario._mpisppy_data.nonant_indices.keys(), domain=pyo.NonNegativeReals)
            #print(f'xdiff index: {scenario._mpisppy_data.nonant_indices.keys()}')
            # def _l1_rule(model, ndn_i):
            #     return model.xdiff_pos[ndn_i] - model.xdiff_neg[ndn_i] == model.nonant_indices[ndn_i]
            # scenario._mpisppy_model.xdiff_split = pyo.Constraint(scenario._mpisppy_data.nonant_indices.keys(), rule=_l1_rule)

            @scenario._mpisppy_model.Constraint(scenario._mpisppy_data.nonant_indices.keys())
            def xdiff_split(model, *ndn_i):
                #print(model, ndn_i)
                return model.xdiff_pos[ndn_i] - model.xdiff_neg[ndn_i] == scenario._mpisppy_data.nonant_indices[ndn_i] - xbars[ndn_i]

            ph_term = 0
            # Dual term (weights W)
            if (add_duals):
                scenario._mpisppy_model.WExpr = pyo.Expression(expr=\
                        sum(scenario._mpisppy_model.W[ndn_i] * xvar \
                            for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items()) )
                ph_term += scenario._mpisppy_model.W_on * scenario._mpisppy_model.WExpr

            # Prox term (quadratic)
            if (add_prox):
                prox_expr = 0.
                for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items():
                    # expand (x - xbar)**2 to (x**2 - 2*xbar*x + xbar**2)
                    # x**2 is the only qradratic term, which might be
                    # dealt with differently depending on user-set options
                    prox_expr += (scenario._mpisppy_model.rho[ndn_i] ) * \
                                 (scenario._mpisppy_model.xdiff_pos[ndn_i] + scenario._mpisppy_model.xdiff_neg[ndn_i])
                scenario._mpisppy_model.ProxExpr = pyo.Expression(expr=prox_expr)
                ph_term += scenario._mpisppy_model.prox_on * scenario._mpisppy_model.ProxExpr

            if (is_min_problem):
                objfct.expr += ph_term
            else:
                objfct.expr -= ph_term