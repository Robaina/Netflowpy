import numpy as np
from gurobipy import Model, GRB
from matplotlib import pyplot as plt

# Semidan Robaina Estevez, 2020

class NetworkFlow:

    """
       Constructs a model as a linear program to be solved by Gurobi
    """

    def __init__(self, S, obj_x=None, x_min=None, x_max=None, v_min=None, v_max=None,
                 xp_min=None, xp_max=None, x_0=None, n_steps=10,
                 x_names=None, v_names=None, v_delta_max=0.1):
        self.S = S
        self.n_steps = n_steps
        self.n_nodes, self.n_edges = S.shape
        self.n_vars = (self.n_steps * (2 * self.n_nodes + 2 * self.n_edges)
                       + self.n_nodes)

        self.obj = np.zeros(self.n_vars)
        if obj_x is not None:
            if x_names is not None:
                obj_x_idx = x_names.index(obj_x)
            else:
                obj_x_idx = obj_x
            for t in range(0, self.n_steps + 1):
                self.obj[t * obj_x_idx] = 1

        if x_min is None:
            x_min = np.zeros(self.n_nodes)
        self.x_min = np.tile(x_min, self.n_steps + 1)
        if x_max is None:
            x_max = 10 * np.ones(self.n_nodes)
        self.x_max = np.tile(x_max, self.n_steps + 1)

        if v_min is None:
            v_min = np.zeros(self.n_edges)
        self.v_min = np.tile(v_min, self.n_steps)
        if v_max is None:
            v_max = 1000 * np.ones(self.n_edges)
        self.v_max = np.tile(v_max, self.n_steps)

        if xp_min is None:
            xp_min = -.01 * np.ones(self.n_nodes)
        else:
            xp_min = xp_min * np.ones(self.n_nodes)
        self.xp_min = np.tile(xp_min, self.n_steps)

        if xp_max is None:
            xp_max = .01 * np.ones(self.n_nodes)
        else:
            xp_max = xp_max * np.ones(self.n_nodes)
        self.xp_max = np.tile(xp_max, self.n_steps)

        if x_0 is not None:
            self.x_min[:self.n_nodes] = x_0
            self.x_max[:self.n_nodes] = x_0

        if x_names is None:
            self.x_names = [f'x_{i}_t{t}'
                            for t in range(self.n_steps + 1)
                            for i in range(self.n_nodes)]
        else:
            self.x_names = [f'{x}_t_{t}'
                           for t in range(self.n_steps + 1)
                           for x in x_names]
            self.original_x_names = x_names
        if v_names is None:
            self.v_names = [f'v_{i}_t{t}'
                           for t in range(self.n_steps)
                           for i in range(self.n_edges)]
        else:
            self.v_names = [f'{v}_t_{t}'
                           for t in range(self.n_steps)
                           for v in v_names]
            self.original_v_names = v_names

        self.xp_names = ['p' + x for x in self.x_names[:-1]]

        self.v_delta_max = v_delta_max

        self._buildLPModel()
        self._convertToGurobiModel()


    def _buildLPModel(self):
        """
           Builds the LP model to be solved by Gurobi
           variables: [x_t, v_t, xp_t, v_delta]
        """
        n_constraints = self.n_steps * 2*self.n_nodes + (self.n_steps-1) * self.n_edges
        A = np.zeros((n_constraints, self.n_vars))

        # Build constraints: S_t * v_t - xp_t = 0
        left_offset_1 = (self.n_steps + 1) * self.n_nodes
        left_offset_2 = left_offset_1 + self.n_steps * self.n_edges
        left_offset_3 = left_offset_2 + self.n_steps * self.n_nodes
        vert_offset_1 = self.n_steps * self.n_nodes
        vert_offset_2 = 2 * vert_offset_1
        I_n = np.eye(self.n_nodes)
        I_e = np.eye(self.n_edges)
        for t in range(0, self.n_steps):

            # Build constraints: S_t * v_t - xp_t = 0
            rows_1 = range(t * self.n_nodes, (t + 1) * self.n_nodes)

            A[rows_1,
              left_offset_1 + t*self.n_edges:left_offset_1 + (t + 1)*self.n_edges] = self.S
            A[rows_1,
              left_offset_2 + t*self.n_nodes:left_offset_2 + (t + 1) * self.n_nodes] = -I_n

            # Build constraints: x_(t+1) - x_t - xp_t = 0
            rows_2 = range(vert_offset_1 + t * self.n_nodes,
                           vert_offset_1 + (t + 1) * self.n_nodes)

            A[rows_2, t * self.n_nodes:(t + 1) * self.n_nodes] = -I_n
            A[rows_2, (t + 1) * self.n_nodes:(t + 2) * self.n_nodes] = I_n
            A[rows_2,
              left_offset_2 + t * self.n_nodes:left_offset_2 + (t + 1) * self.n_nodes] = -I_n

        for t in range(0, self.n_steps - 1):
            # Build constraints: v_(t+1) - v_t - delta = 0
            rows_3 = range(vert_offset_2 + t * self.n_edges,
                           vert_offset_2 + (t + 1) * self.n_edges)

            A[rows_3,
              left_offset_1 + t*self.n_edges:left_offset_1 + (t + 1)*self.n_edges] = -I_e
            A[rows_3,
              left_offset_1 + (t+1)*self.n_edges:left_offset_1 + (t+2)*self.n_edges] = I_e
            A[rows_3,
              left_offset_3 + t*self.n_edges:left_offset_3 + (t + 1)*self.n_edges] = -I_e

        self.A = A

        # Build rhs, lb, ub
        self.rhs = np.zeros(n_constraints)
        v_delta_max = self.v_delta_max * np.ones(self.n_steps * self.n_edges)
        self.lb = np.concatenate((self.x_min, self.v_min, self.xp_min, -v_delta_max))
        self.ub = np.concatenate((self.x_max, self.v_max, self.xp_max, v_delta_max))


    def _convertToGurobiModel(self):
        """
        Convert numpy arrays to gurobi model string
        """
        constraint_sense = ['==' for _ in range(len(self.rhs))]
        model_object = GurobiModel(c=self.obj, A=self.A, b=self.rhs,
                            lb=self.lb, ub=self.ub, modelSense='max',
                            sense=constraint_sense)
        self.model = model_object.construct()


    def changeObjectiveFunction(self, obj_x, sense='max'):
        """
        Build objective function
        obj_x is the x variable index to be optimize across trajectory
        """
        self.obj = np.zeros(self.n_vars)
        for t in range(0, self.n_steps + 1):
            self.obj[t * obj_x] = 1
        self.model = GurobiModel.updateObjective(self.model_object, self.obj, sense=sense)


    def solve(self, verbose=True):
        """
        Solve LP using Gurobi
        """
        if not verbose:
            self.model.setParam('OutputFlag', False)
#         self.model.setParam('FeasibilityTol', 1e-9)
        self.model.optimize()


    def getX(self):
        """
        Retrieve optimal x variables
        """
        try:
            self.model.X
        except Exception:
            raise ValueError('Need to solve model first!')
        x = self.model.X[:(self.n_steps + 1) * self.n_nodes]
        x_values = {}
        for i in range(self.n_nodes):
            series = [x[t * self.n_nodes + i] for t in range(self.n_steps + 1)]
            x_values[self.original_x_names[i]] = series
        return x_values


    def getV(self):
        """
        Retrieve optimal v variables
        """
        try:
            self.model.X
        except Exception:
            raise ValueError('Need to solve model first!')
        v_values = {}
        left_offset = (self.n_steps + 1) * self.n_nodes
        x = self.model.X[left_offset:left_offset + (self.n_steps) * self.n_edges]
        for i in range(self.n_edges):
            series = [x[t * self.n_edges + i] for t in range(self.n_steps)]
            v_values[self.original_v_names[i]] = series
        return v_values


    def getXp(self):
        """
        Retrieve optimal x_p variables
        """
        try:
            self.model.X
        except Exception:
            raise ValueError('Need to solve model first!')
        xp_values = {}
        left_offset = (self.n_steps + 1)*self.n_nodes + self.n_steps*self.n_edges
        x = self.model.X[left_offset:left_offset + (self.n_steps)*self.n_nodes]
        for i in range(self.n_nodes):
            series = [x[t * self.n_nodes + i] for t in range(self.n_steps)]
            xp_values[self.original_x_names[i] + 'p'] = series
        return xp_values


    def getSolutionStatus(self):
        return self.model.Status


    def findAlternativeOptimaBounds(self, vars_to_eval=None):
        """
        Find the minimum and maximum bounds to the optimal X solution found
        previously.
        """
        try:
            self.model.X
        except Exception:
            self.solve()

        if vars_to_eval is None:
            vars_to_eval = self.original_x_names

        model_copy = self.model.copy()
        # Add optimality constraint
        x = model_copy.getVars()
        s = ''
        for j, coeff in enumerate(self.obj):
            if coeff != 0:
                s += 'x[' + str(j) + '] * ' + str(coeff) + '+'
        s = s[:-1]
        s += ' ' + '==' + ' ' + str(self.model.objVal)

        model_copy.addConstr(eval(s))
        model_copy.setParam('OutputFlag', False)
#         model_copy.setParam('FeasibilityTol', 1e-9)
#         model_copy.setParam('OptimalityTol', 1e-9)

        # Loop over variables to evaluate
        vars_bounds = {}
        for var in vars_to_eval:
            vars_bounds[var] = {'min': [], 'max': []}
            time_points = [x for x in self.x_names if var in x]
            time_points_idx = [self.x_names.index(x) for x in time_points]
            c = np.zeros(self.n_vars)
            var_min, var_max = [], []
            for time_point in time_points_idx:
                c = np.zeros(self.n_vars)
                c[time_point] = 1

                model_copy = GurobiModel.updateObjective(model_copy, c, 'min')
                model_copy.update()
                model_copy.optimize()
                var_min.append(model_copy.objVal)

                model_copy = GurobiModel.updateObjective(model_copy, c, 'max')
                model_copy.update()
                model_copy.optimize()
                var_max.append(model_copy.objVal)

            vars_bounds[var]['min'] = var_min
            vars_bounds[var]['max'] = var_max

        self.vars_bounds = vars_bounds


    def sampleAlternativeOptimaSpace(self, n_samples=10):
        """
        Sample the alternative optima space of the LP.
        Returns a sample of trajectories for each of the X variables
        """
        n_x_vars = (self.n_steps + 1) * self.n_nodes
        n_v_vars = self.n_steps * self.n_edges
        n_xp_vars = self.n_steps * self.n_nodes
        max_epsilon = 1e8

        # Add new epsilon variables: e = x - xrand
        B = np.hstack((self.A, np.zeros((self.A.shape[0], n_x_vars))))
        # Add constraint x - e = xrand
        C = np.hstack((np.eye(n_x_vars),
                       np.zeros((n_x_vars, 2*n_v_vars + n_xp_vars)),
                      -np.eye(n_x_vars)))
        D = np.vstack((B, C))

        c_obj = np.ones(D.shape[1]) # dummy objective
        rhs = np.zeros(D.shape[0])
        lb = np.concatenate((self.lb, -max_epsilon * np.ones(n_x_vars)))
        ub = np.concatenate((self.ub, max_epsilon * np.ones(n_x_vars)))
        sense = ['==' for _ in range(len(rhs))]

        # Build gurobi model
        model_object = GurobiModel(c=c_obj, A=D, b=rhs,
                            lb=lb, ub=ub, modelSense='min',
                            sense=sense)
        model = model_object.construct()
#         model.setParam('FeasibilityTol', 1e-9)
#         model.setParam('OptimalityTol', 1e-9)

        # Add optimality constraint
        x = model.getVars()
        s = ''
        for j, coeff in enumerate(self.obj):
            if coeff != 0:
                s += f'x[{j}] * {coeff} +'
        s = s[:-1]
        s += ' ' + '==' + ' ' + str(self.model.objVal)

        # Add quadratic objective: e_0^2 + e_1^2 + ... e_n^2
        obj_str = ''
        for i in range(n_x_vars + 2 * n_v_vars + n_xp_vars,
                       2 * n_x_vars + 2*n_v_vars + n_xp_vars):
            obj_str += f'x[{i}]*x[{i}] +'
        obj_str = obj_str[:-1]

        model.setObjective(eval(obj_str), GRB.MINIMIZE)
        model.addConstr(eval(s))
        model.setParam('OutputFlag', False)
        model.update()

        sampled_x = {k: [] for k in self.original_x_names}
        min_x, max_x = min(self.lb), max(self.ub)
        # Addd last value to rhs (optimal value)
        rhs = np.append(rhs, self.model.objval)

        # Begin sampling
        for n in range(n_samples):

            # Generate random x vector
            x_rand = (max_x - min_x) * np.random.rand(n_x_vars) + min_x
            rhs[-(n_x_vars+1):-1] = x_rand

            # Update rhs
            GurobiModel.updateRHS(model, rhs)
            model.update()
            # Solve
            model.optimize()
#             if model.Status != 2:
#                 print('Not optimal')
            # Retrieve solutions
            x = model.X
            for i in range(self.n_nodes):
                series = [x[t * self.n_nodes + i] for t in range(self.n_steps + 1)]
                sampled_x[self.original_x_names[i]].append(series)

        self.vars_AO_sample = sampled_x


    def _computeAOSampleMean(self, vars_to_eval):
        """
        Compute mean trajectory of alternative optima sample
        """
        mean_trajectories = {}
        for var in vars_to_eval:
            var_sample = np.array(self.vars_AO_sample[var])
            mean_trajectories[var] = var_sample.mean(0)
        return mean_trajectories


    def plotXSolution(self, var_to_plot):
        """
        Plot X trajectories and alternative optima bounds if present
        """
        plt.figure(figsize=(14, 8))
        plt.title(f'Variable: {var_to_plot}')
        plt.xlabel('t')
        plt.ylabel('X')

        if hasattr(self, 'vars_AO_sample'):
            samples = self.vars_AO_sample[var_to_plot]
            samples_mean = self._computeAOSampleMean(var_to_plot)
            for data in samples:
                plt.plot(range(self.n_steps + 1), data, color='lightgrey')
            plt.plot(range(self.n_steps + 1), samples_mean[var_to_plot], color='grey')
        else:
            plt.plot(range(self.n_steps + 1), self.getX()[var_to_plot], color='grey')

        if hasattr(self, 'vars_bounds') and var_to_plot in self.vars_bounds.keys():
            plt.plot(range(self.n_steps + 1), self.vars_bounds[var_to_plot]['min'],
                     color='blue')
            plt.plot(range(self.n_steps + 1), self.vars_bounds[var_to_plot]['max'],
                     color='red')
        plt.show()


    def plotVSolution(self, var_to_plot):
        """
        Plot V trajectories and alternative optima bounds if present
        """
        plt.figure(figsize=(14, 8))
        plt.title(f'Variable: {var_to_plot}')
        plt.xlabel('t')
        plt.ylabel('V')
        plt.plot(range(self.n_steps), self.getV()[var_to_plot], color='blue')
        plt.show()



class GurobiModel:
    """
    Constructs a gurobipy model from a matrix formulation. Currently only LPs
    and MILPs are covered, i.e., the general optimization problem:
                  min   cx
                  s.t.
                      Ax <= b (>=, ==)
                      lb <= x <= ub
    Arguments:
    ---------
    c: 1D array, the objective vector
    A: 2D array, the constraint matrix
    lb, ub: 1D array, lower and upper bounds for the variables (default, 0 to Inf)
    modelSense: str, the optimization sense, 'min' or 'max'
    sense: array-like of str, the constraints sense: <=, == or >= (default <=)
    binaryVariables: array (optional), column indices of A corresponding
                     to binary variables (default continuous)
    variableNames: array-like of str (optional), the names of the variables
    modelName: str, (optional) the name of the gurobi model.
    """

    def __init__(self, c, A, b, lb=0, ub=GRB.INFINITY, modelSense='min',
                 sense=None, binaryVariables=None,
                 variableNames=None, modelName='model'):
        self.obj = c
        self.A = A
        self.rhs = b
        self.ub = ub
        self.lb = lb
        self.modelSense = modelSense

        if modelName is None:
            self.modelName = 'model'
        else:
            self.modelName = modelName

        self.nConst, self.nVars = np.shape(A)
        if sense is None:
            self.sense = ['<=' for _ in range(self.nConst)]
        else:
            self.sense = sense
        self.varType = np.array(['C' for _ in range(self.nVars)])
        if binaryVariables is not None:
            self.varType[binaryVariables] = 'B'
        self.binaryVariables = binaryVariables

        if variableNames is None:
            self.varNames = ['x' + str(n) for n in range(self.nVars)]
        else:
            self.varNames = variableNames

    def construct(self):
        """
        Builds a gurobipy model object
        """
        model = Model(self.modelName)

        x = model.addVars(range(self.nVars), lb=self.lb, ub=self.ub,
                          obj=self.obj, vtype=self.varType, name=self.varNames)

        for i, row in enumerate(self.A):
            s = ''
            for j, coeff in enumerate(row):
                if coeff != 0:
                    s += 'x[' + str(j) + '] * ' + str(coeff) + '+'
            s = s[:-1]
            s += ' ' + self.sense[i] + ' ' + str(self.rhs[i])
            model.addConstr(eval(s))

        model = self.updateObjective(model, self.obj, self.modelSense)
        return model

    @staticmethod
    def updateObjective(model, c, sense='min'):
        """
        Updates the objective vector c of a linear model
        """
        if sense.lower() in 'min':
            objsense = GRB.MINIMIZE
        else:
            objsense = GRB.MAXIMIZE
        model.update()

        x = model.getVars()
        o = ''

        for i, coeff in enumerate(c):
            #if coeff != 0:
            o += 'x[' + str(i) + '] * ' + str(coeff) + '+'
        o = o[:-1]

        model.setObjective(eval(o), objsense)
        model.update()
        return model

    @staticmethod
    def updateRHS(model, b):
        """
        Updates the right-hand-side vector b of the model constraints
        """
        model.update()
        Constrs = model.getConstrs()
        for i, constr in enumerate(Constrs):
            constr.setAttr('rhs', b[i])

        model.update()
        return model
