# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

from math import isclose
from pyomo.environ import value
from pyomo.core.expr.numeric_expr import LinearExpression
import numpy as np

# helpers for distance from y = x**2
def _f(val, x_pnt, y_pnt):
    return (( val - x_pnt )**2 + ( val**2 - y_pnt )**2)/2.
def _df(val, x_pnt, y_pnt):
    #return 2*(val - x_pnt) + 4*(val**2 - y_pnt)*val
    return val*(1 - 2*y_pnt + 2*val*val) - x_pnt
def _d2f(val, x_pnt, y_pnt):
    return 1 + 6*val*val - 2*y_pnt

def _newton_step(val, x_pnt, y_pnt):
    return val - _df(val, x_pnt, y_pnt) / _d2f(val, x_pnt, y_pnt)

class ProxApproxManager:
    __slots__ = ()

    def __new__(cls, xvar, xvarsqrd, xbar, xsqbar, xsqvar_cuts, ndn_i):
        if xvar.is_integer():
            return ProxApproxManagerDiscrete(xvar, xvarsqrd, xbar, xsqbar, xsqvar_cuts, ndn_i)
        else:
            return ProxApproxManagerContinuous(xvar, xvarsqrd, xbar, xsqbar, xsqvar_cuts, ndn_i)

class _ProxApproxManager:
    '''
    A helper class to manage proximal approximations
    '''
    __slots__ = ()

    def __init__(self, xvar, xvarsqrd, xbar, xsqbar, xsqvar_cuts, ndn_i):
        self.xvar = xvar
        self.xvarsqrd = xvarsqrd
        self.xbar = xbar
        self.xsqbar = xsqbar
        self.var_index = ndn_i
        self.cuts = xsqvar_cuts
        self.cut_index = 0
        self._store_bounds()
        self._lin_points = {}
        self._xbar_history = {}

    def _store_bounds(self):
        if self.xvar.lb is None:
            self.lb = -float("inf")
        else:
            self.lb = self.xvar.lb
        if self.xvar.ub is None:
            self.ub = float("inf")
        else:
            self.ub = self.xvar.ub

    def add_cut(self, val, persistent_solver=None):
        '''
        create a cut at val
        '''
        pass

    def maintain_cuts(self, persistent_solver=None):
        # TODO: move option to phbase
        max_num_cuts = 20

        if len(self._lin_points) > max_num_cuts:
            num_del = len(self._lin_points) - max_num_cuts
            x_arr = np.fromiter(self._lin_points.values(), dtype=float)
            x_idx = np.fromiter(self._lin_points.keys(), dtype=int)
            curr_xbar = self.xbar.value
            # sort lin points by abs distance to xbar
            #print(f'xbar: {curr_xbar}, x_arr: {x_arr}, x_idx: {x_idx}')
            order = np.argsort(np.abs(x_arr - curr_xbar))
            x_arr = x_arr[order]
            x_idx = x_idx[order]
            #print(f'xbar: {curr_xbar}, x_arr: {x_arr}, x_idx: {x_idx}')
            for k in range(1, num_del + 1):
                del_idx = x_idx[-k]
                #print(f"deleting cut idx {del_idx}")
                self._lin_points.pop(del_idx)
                # TODO: always
                if persistent_solver is not None:
                    persistent_solver.remove_constraint(self.cuts[self.var_index, del_idx])
                
                del self.cuts[self.var_index, del_idx]
                #del self.cuts[self.var_index, del_idx]
                #print(f"deleting cut for {self.xvar.name} at {del_val}")


    def check_tol_add_cut(self, tolerance, persistent_solver=None):
        '''
        add a cut if the tolerance is not satified
        '''
        x_pnt = self.xvar.value
        x_bar = self.xbar.value
        xb_variance = np.maximum(self.xsqbar.value - x_bar * x_bar, 0)
        rel_sd = np.abs(np.sqrt(xb_variance)/(x_bar + 1e-6))
        rel_sd = np.maximum(rel_sd, 1e-6)
        #rel_sd = 1
        y_pnt = self.xvarsqrd.value
        f_val = x_pnt**2
        lb = self.xvar.lb
        ub = self.xvar.ub

        tol = np.maximum(x_pnt, 1e-6) * tolerance
        tol *= rel_sd

        #print(f"y-distance: {actual_val - measured_val})")
        if y_pnt is None:
            self.add_cut(x_pnt, persistent_solver)
            self.add_cut(x_bar + xb_variance, persistent_solver)
            self.add_cut(x_bar - xb_variance, persistent_solver)
            #if lb is not None:
            #    self.add_cut(lb, persistent_solver)
            #if ub is not None:
            #    self.add_cut(ub, persistent_solver)
            
            if not isclose(x_pnt, x_bar, abs_tol=1e-6):
                self.add_cut(2*x_bar - x_pnt, persistent_solver)
            return True

        #print(f'tol: {tol}')
        if (f_val - y_pnt) > tol:
            '''
            In this case, we project the point x_pnt, y_pnt onto
            the curve y = x**2 by finding the minimum distance
            between y = x**2 and x_pnt, y_pnt.

            This involves solving a cubic equation, so instead
            we start at x_pnt, y_pnt and run newtons algorithm
            to get an approximate good-enough solution.
            '''
            this_val = x_pnt
            #print(f"initial distance: {_f(this_val, x_pnt, y_pnt)**(0.5)}")
            #print(f"this_val: {this_val}")
            # self.add_cut(this_val, persistent_solver)
            # if not isclose(this_val, x_bar, abs_tol=1e-6):
            #     # TODO: try without this cut
            #     self.add_cut(2*x_bar - this_val, persistent_solver)
            next_val = _newton_step(this_val, x_pnt, y_pnt)
            #next_val = this_val
            while not isclose(this_val, next_val, rel_tol=1e-6, abs_tol=1e-6):
                #print(f"newton step distance: {_f(next_val, x_pnt, y_pnt)**(0.5)}")
                #print(f"next_val: {next_val}")
                this_val = next_val
                next_val = _newton_step(this_val, x_pnt, y_pnt)
            
            # if not isclose(next_val, this_val, abs_tol=1e-6):
            self.add_cut(next_val, persistent_solver)
            # TODO: change tolerance - make relative?
            if not isclose(next_val, x_bar, abs_tol=1e-6, ):
                # TODO: try without this cut
                self.add_cut(2*x_bar - next_val, persistent_solver)
                #self.add_cut(x_bar, persistent_solver)
            #self.maintain_cuts(persistent_solver)
            return True
        return False
    
    # def check_tol_add_cut(self, tolerance, persistent_solver=None):
    #     '''
    #     add a cut if the tolerance is not satified
    #     '''
    #     x_pnt = self.xvar.value
    #     x_bar = self.xbar.value
    #     xb_variance = self.xsqbar.value - x_bar * x_bar
    #     xb_sd = np.sqrt(xb_variance)
    #     rel_sd = np.sqrt(xb_variance)/x_bar
    #     #rel_sd = 1
    #     y_pnt = self.xvarsqrd.value
    #     f_val = x_pnt**2
    #     lb = self.xvar.lb
    #     ub = self.xvar.ub

    #     if y_pnt is None:
    #         self.add_cut(x_pnt, persistent_solver)
    #         self.add_cut(x_bar + xb_sd)
    #         self.add_cut(x_bar - xb_sd)
    #         return True
    #     #print(f"y-distance: {actual_val - measured_val})")

    #     if (f_val - y_pnt) > (tolerance):
    #         this_val = x_pnt
    #         next_val = _newton_step(this_val, x_pnt, y_pnt)
    #         while not isclose(this_val, next_val, rel_tol=1e-6, abs_tol=1e-6):
    #             this_val = next_val
    #             next_val = _newton_step(this_val, x_pnt, y_pnt)
    #         self.add_cut(next_val, persistent_solver)
    #         self.add_cut(x_bar + xb_sd)
    #         self.add_cut(x_bar - xb_sd)
    #         return True

    #     return False

        

class ProxApproxManagerContinuous(_ProxApproxManager):

    def add_cut(self, val, persistent_solver=None):
        '''
        create a cut at val using a taylor approximation
        '''
        # handled by bound
        if val == 0:
            return 0

        self._lin_points[self.cut_index] = val
        if self.xbar.value in self._xbar_history:
            self._xbar_history[self.xbar.value].append(self.cut_index)
        else:
            self._xbar_history[self.xbar.value] = [self.cut_index]
        # f'(a) = 2*val
        # f(a) - f'(a)a = val*val - 2*val*val
        f_p_a = 2*val
        const = -(val*val)

        ## f(x) >= f(a) + f'(a)(x - a)
        ## f(x) >= f'(a) x + (f(a) - f'(a)a)
        ## (0 , f(x) - f'(a) x - (f(a) - f'(a)a) , None)
        expr = LinearExpression( linear_coefs=[1, -f_p_a],
                                 linear_vars=[self.xvarsqrd, self.xvar],
                                 constant=-const )
        self.cuts[self.var_index, self.cut_index] = (0, expr, None)
        if persistent_solver is not None:
            persistent_solver.add_constraint(self.cuts[self.var_index, self.cut_index])
        self.cut_index += 1
        #print(f"added continuous cut for {self.xvar.name} at {val}, lb: {self.xvar.lb}, ub: {self.xvar.ub}")

        return 1
    
    def plot(self, plot_range=(-50,50), plot_points=100, folder='prox_approx_plots', filename=''):
        import matplotlib.pyplot as plt
        def lin_xsq(x, lin_points):
            lin_values = []
            for lin_point in lin_points:
                lin_values.append(2*lin_point*x - lin_point**2)
            return max(lin_values)
        
        f1, ax1 = plt.subplots()
        f2, ax2 = plt.subplots()
        x1 = np.linspace(*plot_range, plot_points)
        x2 = np.linspace(*plot_range, plot_points)
        y1 = x1**2
        y2 = x2**2
        #ax1.plot(x1, y1, label='(x-xb)^2')
        ax2.plot(x2, y2, label='x^2')
        for xbar, cidx_list in self._xbar_history.items():
            max_cidx = max(cidx_list)
            lin_points = []
            for ixb, icidx_l in self._xbar_history.items():
                prev_cidx = [c for c in icidx_l if c <= max_cidx]
                lin_points += [self._lin_points[c] for c in prev_cidx]
                if ixb == xbar:
                    curr_lin_points = [self._lin_points[c] for c in cidx_list]
            
            y_lin = [lin_xsq(xp, lin_points) for xp in x2]
            l = ax2.plot(x2, y_lin, label=f'lin. for xb={xbar}')
            ax2.scatter(curr_lin_points, [xp ** 2 for xp in curr_lin_points], color=l[0].get_color(), marker='x')
            ax2.scatter(xbar, xbar**2, color=l[0].get_color(), marker='o')

            d_x_x2_lin = [lin_xsq(xp, lin_points) - 2 * xp * xbar + xbar ** 2 for xp in x1]
            l1 = ax1.plot(x1, d_x_x2_lin, label=f'lin. for (x-{xbar})^2')
            ax1.plot(x1, (x1 -xbar)**2, color=l1[0].get_color())
        
        ax1.legend()
        ax1.set_xlabel('x-xb')
        ax1.set_title('Approximations of (x-xb)^2 ')
        f1.savefig(f'{folder}/prox_term_{filename}', dpi=400)
        plt.close(f1)

        ax2.legend()
        ax2.set_xlabel('x')
        ax2.set_title('Approximations of x^2 ')
        f2.savefig(f'{folder}/x^2_term_{filename}', dpi=400)
        plt.close(f2)

        return None


def _compute_mb(val):
    ## [(n+1)^2 - n^2] = 2n+1
    ## [(n+1) - n] = 1
    ## -> m = 2n+1
    m = 2*val+1

    ## b = n^2 - (2n+1)*n
    ## = -n^2 - n
    ## = -n (n+1)
    b = -val*(val+1)
    return m,b

class ProxApproxManagerDiscrete(_ProxApproxManager):

    def add_cut(self, val, persistent_solver=None):
        '''
        create up to two cuts at val, exploiting integrality
        '''
        val = int(round(val))

        ## cuts are indexed by the x-value to the right
        ## e.g., the cut for (2,3) is indexed by 3
        ##       the cut for (-2,-1) is indexed by -1
        cuts_added = 0

        ## So, a cut to the RIGHT of the point 3 is the cut for (3,4),
        ## which is indexed by 4
        if (*self.var_index, val+1) not in self.cuts and val < self.ub:
            m,b = _compute_mb(val)
            expr = LinearExpression( linear_coefs=[1, -m],
                                     linear_vars=[self.xvarsqrd, self.xvar],
                                     constant=-b )
            #print(f"adding cut for {(val, val+1)}")
            self.cuts[self.var_index, val+1] = (0, expr, None)
            if persistent_solver is not None:
                persistent_solver.add_constraint(self.cuts[self.var_index, val+1])
            cuts_added += 1
            # TODO: could also use val+1 or val + 0.5
            self._lin_points[self.cut_index] = val + 0.5
            self.cut_index += 1

        ## Similarly, a cut to the LEFT of the point 3 is the cut for (2,3),
        ## which is indexed by 3
        if (*self.var_index, val) not in self.cuts and val > self.lb:
            m,b = _compute_mb(val-1)
            expr = LinearExpression( linear_coefs=[1, -m],
                                     linear_vars=[self.xvarsqrd, self.xvar],
                                     constant=-b )
            #print(f"adding cut for {(val-1, val)}")
            self.cuts[self.var_index, val] = (0, expr, None)
            if persistent_solver is not None:
                persistent_solver.add_constraint(self.cuts[self.var_index, val])
            cuts_added += 1
            self._lin_points[self.cut_index] = val - 0.5
            self.cut_index += 1
        #print(f"added {cuts_added} integer cut(s) for {self.xvar.name} at {val}, lb: {self.xvar.lb}, ub: {self.xvar.ub}")

        return cuts_added

if __name__ == '__main__':
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    bounds = (-100, 100)
    m.x = pyo.Var(bounds = bounds)
    #m.x = pyo.Var(within=pyo.Integers, bounds = bounds)
    m.xsqrd = pyo.Var(within=pyo.NonNegativeReals)

    xbars = [50, 10, -5]

    xsqbars = [60, 8, -4.5]

    m.xsqbar = pyo.Param(initialize=100, mutable=True)

    m.zero = pyo.Param(initialize=100, mutable=True)
    ## ( x - zero )^2 = x^2 - 2 x zero + zero^2
    m.obj = pyo.Objective( expr = m.xsqrd - 2*m.zero*m.x + m.zero**2 )

    m.xsqrdobj = pyo.Constraint([0], pyo.Integers)

    s = pyo.SolverFactory('gurobi_persistent')
    prox_manager = ProxApproxManager(m.x, m.xsqrd, m.zero, m.xsqbar, m.xsqrdobj, 0)
    s.set_instance(m)
    m.pprint()
    new_cuts = True
    iter_cnt = 0
    for i, xb in enumerate(xbars):
        s.solve(m,tee=False)
        print(f"x: {pyo.value(m.x):.2e}, obj: {pyo.value(m.obj):.2e}")
        new_cuts = prox_manager.check_tol_add_cut(1e-1, persistent_solver=s)
        m.zero = xb
        m.xsqbar = xsqbars[i]
        #m.pprint()
        iter_cnt += 1
    
    prox_manager.plot(plot_range=(-200,200),plot_points=100)

    print(f"cuts: {len(m.xsqrdobj)}, iters: {iter_cnt}")
