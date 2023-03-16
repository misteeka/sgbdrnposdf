
import collections
import copy
from collections.abc import Iterable
from typing import List

import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.optimize as sco

# Выпуклая оптимизация с cvxpy
class Optimizer():
    def __init__(
        self,
        expected_returns,
        cov_matrix,
        weight_bounds=(0, 1),
        solver_options=None,
    ):  
        self.cov_matrix = cov_matrix
        self.expected_returns = expected_returns
        self._max_return_value = None

        if self.expected_returns is None:
            num_assets = len(cov_matrix)
        else:
            num_assets = len(expected_returns)

        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:  # int labels
            tickers = list(range(num_assets))
        self.tickers = tickers

        # Optimization variables
        self.n_assets = len(tickers)
        self._w = cp.Variable(len(tickers))
        self._objective = None
        self._additional_objectives = []
        self._constraints = []
        self._lower_bounds = None
        self._upper_bounds = None
        self._opt = None
        self._solver = None
        self._solver_options = solver_options if solver_options else {}
        self._map_bounds_to_constraints(weight_bounds)

    def clean_weights(self, cutoff=0.01, rounding=5):
        clean_weights = self.weights.copy()
        clean_weights[np.abs(clean_weights) < cutoff] = 0
        if rounding is not None:
            clean_weights = np.round(clean_weights, rounding)

        return self._make_output_weights(clean_weights)

    def _make_output_weights(self, weights=None):
        if weights is None:
            weights = self.weights
        return collections.OrderedDict(zip(self.tickers, weights))

    def _solve_cvxpy_opt_problem(self):
        try:
            if self._opt is None:
                self._opt = cp.Problem(cp.Minimize(self._objective), self._constraints)
                self._initial_objective = self._objective.id
                self._initial_constraint_ids = {const.id for const in self._constraints}
            self._opt.solve(
                solver=self._solver, verbose=False, **self._solver_options
            )

        except (TypeError, cp.DCPError) as e:
            raise e
        self.weights = self._w.value.round(16) + 0.0  # +0.0 removes signed zero
        return self._make_output_weights()

    def add_constraint(self, new_constraint):
        self._constraints.append(new_constraint(self._w))

    def min_volatility(self):
        self._objective = portfolio_variance(
            self._w, self.cov_matrix
        )
        for obj in self._additional_objectives:
            self._objective += obj

        self.add_constraint(lambda w: cp.sum(w) == 1)
        return self._solve_cvxpy_opt_problem()

    def _max_return(self, return_value=True):
        self._objective = portfolio_return(
            self._w, self.expected_returns
        )

        self.add_constraint(lambda w: cp.sum(w) == 1)

        res = self._solve_cvxpy_opt_problem()

        if return_value:
            return -self._opt.value
        else:
            return res

    def is_parameter_defined(self, parameter_name: str) -> bool:
        is_defined = False
        objective_and_constraints = (
            self._constraints + [self._objective]
            if self._objective is not None
            else self._constraints
        )
        for expr in objective_and_constraints:
            params = [
                arg for arg in _get_all_args(expr) if isinstance(arg, cp.Parameter)
            ]
            for param in params:
                if param.name() == parameter_name and not is_defined:
                    is_defined = True
        return is_defined

    def efficient_risk(self, target_volatility):
        if not isinstance(target_volatility, (float, int)) or target_volatility < 0:
            raise ValueError("Заданный риск должен быть float и >= 0")
 
        global_min_volatility = np.sqrt(1 / np.sum(np.linalg.pinv(self.cov_matrix)))

        if target_volatility < global_min_volatility:
            raise ValueError("Минимальный риск равен {:.3f}. Используйте более высокий заданный риск".format(global_min_volatility))

        update_existing_parameter = self.is_parameter_defined("target_variance")
        if update_existing_parameter:
            self.update_parameter_value("target_variance", target_volatility**2)
        else:
            self._objective = portfolio_return(
                self._w, self.expected_returns
            )
            variance = portfolio_variance(self._w, self.cov_matrix)

            for obj in self._additional_objectives:
                self._objective += obj

            target_variance = cp.Parameter(
                name="target_variance", value=target_volatility**2, nonneg=True
            )
            self.add_constraint(lambda _: variance <= target_variance)
            self.add_constraint(lambda w: cp.sum(w) == 1)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return):
        if not isinstance(target_return, float):
            raise ValueError("Заданная доходность должна быть float")
        if not self._max_return_value:
            a = self.deepcopy()
            self._max_return_value = a._max_return()
        if target_return > self._max_return_value:
            raise ValueError("Заданная доходность должна быть меньше чем максимально возможная")

        update_existing_parameter = self.is_parameter_defined("target_return")
        if update_existing_parameter:
            self._validate_market_neutral()
            self.update_parameter_value("target_return", target_return)
        else:
            self._objective = portfolio_variance(
                self._w, self.cov_matrix
            )
            ret = portfolio_return(
                self._w, self.expected_returns, negative=False
            )

            for obj in self._additional_objectives:
                self._objective += obj

            target_return_par = cp.Parameter(name="target_return", value=target_return)
            self.add_constraint(lambda _: ret >= target_return_par)
            self.add_constraint(lambda w: cp.sum(w) == 1)
        return self._solve_cvxpy_opt_problem()

    def _map_bounds_to_constraints(self, test_bounds):
        # If it is a collection with the right length, assume they are all bounds.
        if len(test_bounds) == self.n_assets and not isinstance(
            test_bounds[0], (float, int)
        ):
            bounds = np.array(test_bounds, dtype=float)
            self._lower_bounds = np.nan_to_num(bounds[:, 0], nan=-np.inf)
            self._upper_bounds = np.nan_to_num(bounds[:, 1], nan=np.inf)
        else:
            lower, upper = test_bounds

            # Replace None values with the appropriate +/- 1
            if np.isscalar(lower) or lower is None:
                lower = -1 if lower is None else lower
                self._lower_bounds = np.array([lower] * self.n_assets)
                upper = 1 if upper is None else upper
                self._upper_bounds = np.array([upper] * self.n_assets)
            else:
                self._lower_bounds = np.nan_to_num(lower, nan=-1)
                self._upper_bounds = np.nan_to_num(upper, nan=1)

        self.add_constraint(lambda w: w >= self._lower_bounds)
        self.add_constraint(lambda w: w <= self._upper_bounds)

    def deepcopy(self):
        self_copy = copy.copy(self)
        self_copy._additional_objectives = [
            copy.copy(obj) for obj in self_copy._additional_objectives
        ]
        self_copy._constraints = [copy.copy(con) for con in self_copy._constraints]
        return self_copy
    
    def portfolio_performance(self, risk_free_rate=0.02):
        if isinstance(self.weights, dict):
            if isinstance(self.expected_returns, pd.Series):
                tickers = list(self.expected_returns.index)
            elif isinstance(self.cov_matrix, pd.DataFrame):
                tickers = list(self.cov_matrix.columns)
            else:
                tickers = list(range(len(self.expected_returns)))
            new_weights = np.zeros(len(tickers))

            for i, k in enumerate(tickers):
                if k in self.weights:
                    new_weights[i] = self.weights[k]
        else:
            new_weights = np.asarray(self.weights)

        sigma = np.sqrt(portfolio_variance(new_weights, self.cov_matrix))

        if self.expected_returns is not None:
            mu = portfolio_return(new_weights, self.expected_returns, negative=False)
            return mu, sigma
        else:
            return None, sigma   

def _get_all_args(expression: cp.Expression) -> List[cp.Expression]:
    if expression.args == []:
        return [expression]
    else:
        return list(_flatten([_get_all_args(arg) for arg in expression.args]))

def _flatten(l: Iterable) -> Iterable:
    # Helper method to flatten an iterable
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el

def _objective_value(w, obj):
    if isinstance(w, np.ndarray):
        if np.isscalar(obj):
            return obj
        elif np.isscalar(obj.value):
            return obj.value
        else:
            return obj.value.item()
    else:
        return obj

def portfolio_variance(w, cov_matrix):
    variance = cp.quad_form(w, cov_matrix)
    return _objective_value(w, variance)

def portfolio_return(w, expected_returns, negative=True):
    sign = -1 if negative else 1
    mu = w @ expected_returns
    return _objective_value(w, sign * mu)

