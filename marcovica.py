import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cp
import pandas_datareader as web
from matplotlib.ticker import FuncFormatter
from pypfopt.efficient_frontier import EfficientFrontier 
from pypfopt import risk_models, objective_functions
from pypfopt import expected_returns
from pypfopt.cla import CLA
import pypfopt.plotting as pplt
from matplotlib.ticker import FuncFormatter

def getStocksData(start, end):
    tickers = ['LKOH.ME','GMKN.ME', 'DSKY.ME', 'NKNC.ME', 'MTSS.ME', 'IRAO.ME', 'SBER.ME', 'AFLT.ME']
    df_stocks= yf.download(tickers, start=start, end=end)['Adj Close']
    #df_stocks.head()
    nullin_df = pd.DataFrame(df_stocks,columns=tickers)
    nullin_df.isnull().sum()
    return df_stocks

# Годовая доходность
def get_mu(stocks):
    frequency = 252
    if not isinstance(prices, pd.DataFrame):
        print("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices)

    if compounding:
        return (1 + returns).prod() ** (frequency / returns.count()) - 1
    else:
        return returns.mean() * frequency

# Cov
def get_sigma(stocks):
    return risk_models.sample_cov(stocks)

def get_max_sharpe(stocks):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = EfficientFrontier(mu, sigma, weight_bounds=(0,1)) 
    maxsh=ef.max_sharpe()
    return ef

def getMinRisk(stocks):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = EfficientFrontier(mu, sigma, weight_bounds=(0,1)) 
    ef.min_volatility()
    return ef.portfolio_performance()[1]

def getMaxReturn(stocks):
    return max(get_mu(stocks).values)



class EfficientFrontier():

    def __init__(
        self,
        expected_returns,
        cov_matrix,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        # Inputs
        self.cov_matrix = EfficientFrontier._validate_cov_matrix(cov_matrix)
        self.expected_returns = EfficientFrontier._validate_expected_returns(
            expected_returns
        )
        self._max_return_value = None
        self._market_neutral = None

        if self.expected_returns is None:
            num_assets = len(cov_matrix)
        else:
            num_assets = len(expected_returns)

        # Labels
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:  # use integer labels
            tickers = list(range(num_assets))

        if expected_returns is not None and cov_matrix is not None:
            if cov_matrix.shape != (num_assets, num_assets):
                raise ValueError("Covariance matrix does not match expected returns")

        super().__init__(
            len(tickers),
            tickers,
            weight_bounds,
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )

    @staticmethod
    def _validate_expected_returns(expected_returns):
        if expected_returns is None:
            return None
        elif isinstance(expected_returns, pd.Series):
            return expected_returns.values
        elif isinstance(expected_returns, list):
            return np.array(expected_returns)
        elif isinstance(expected_returns, np.ndarray):
            return expected_returns.ravel()
        else:
            raise TypeError("expected_returns is not a series, list or array")

    @staticmethod
    def _validate_cov_matrix(cov_matrix):
        if cov_matrix is None:
            raise ValueError("cov_matrix must be provided")
        elif isinstance(cov_matrix, pd.DataFrame):
            return cov_matrix.values
        elif isinstance(cov_matrix, np.ndarray):
            return cov_matrix
        else:
            raise TypeError("cov_matrix is not a dataframe or array")
        
    def _make_weight_sum_constraint(self):
        self.add_constraint(lambda w: cp.sum(w) == 1)

    def min_volatility(self):
        self._objective = objective_functions.portfolio_variance(
            self._w, self.cov_matrix
        )
        for obj in self._additional_objectives:
            self._objective += obj

        self.add_constraint(lambda w: cp.sum(w) == 1)
        return self._solve_cvxpy_opt_problem()

    def _max_return(self, return_value=True):
        if self.expected_returns is None:
            raise ValueError("no expected returns provided")

        self._objective = objective_functions.portfolio_return(
            self._w, self.expected_returns
        )

        self.add_constraint(lambda w: cp.sum(w) == 1)

        res = self._solve_cvxpy_opt_problem()

        if return_value:
            return -self._opt.value
        else:
            return res
        
    # Максимизация доходности для заданного риска
    def efficient_risk(self, target_volatility):
        if not isinstance(target_volatility, (float, int)) or target_volatility < 0:
            raise ValueError("target_volatility should be a positive float")

        global_min_volatility = np.sqrt(1 / np.sum(np.linalg.pinv(self.cov_matrix)))

        if target_volatility < global_min_volatility:
            raise ValueError(
                "The minimum volatility is {:.3f}. Please use a higher target_volatility".format(
                    global_min_volatility
                )
            )

        update_existing_parameter = self.is_parameter_defined("target_variance")
        if update_existing_parameter:
            self.update_parameter_value("target_variance", target_volatility**2)
        else:
            self._objective = objective_functions.portfolio_return(
                self._w, self.expected_returns
            )
            variance = objective_functions.portfolio_variance(self._w, self.cov_matrix)

            for obj in self._additional_objectives:
                self._objective += obj

            target_variance = cp.Parameter(
                name="target_variance", value=target_volatility**2, nonneg=True
            )
            self.add_constraint(lambda _: variance <= target_variance)
            self._make_weight_sum_constraint()
        return self._solve_cvxpy_opt_problem()

    # Минимизация риска при заданной доходности
    def efficient_return(self, target_return):
        if not isinstance(target_return, float):
            raise ValueError("target_return should be a float")
        if not self._max_return_value:
            a = self.deepcopy()
            self._max_return_value = a._max_return()
        if target_return > self._max_return_value:
            raise ValueError(
                "target_return must be lower than the maximum possible return"
            )

        update_existing_parameter = self.is_parameter_defined("target_return")
        if update_existing_parameter:
            self.update_parameter_value("target_return", target_return)
        else:
            self._objective = objective_functions.portfolio_variance(
                self._w, self.cov_matrix
            )
            ret = objective_functions.portfolio_return(
                self._w, self.expected_returns, negative=False
            )

            for obj in self._additional_objectives:
                self._objective += obj

            target_return_par = cp.Parameter(name="target_return", value=target_return)
            self.add_constraint(lambda _: ret >= target_return_par)
            self._make_weight_sum_constraint()
        return self._solve_cvxpy_opt_problem()

    # Расчет доходности, риска и к. Шарпа
    def portfolio_performance(self):
        if self._risk_free_rate is not None:
            risk_free_rate = self._risk_free_rate
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(self.cov_matrix, pd.DataFrame):
            tickers = list(self.cov_matrix.columns)
        else:
            tickers = list(range(len(expected_returns)))
        new_weights = np.zeros(len(tickers))

        for i, k in enumerate(tickers):
            if k in self.weights:
                new_weights[i] = self.weights[k]
        if new_weights.sum() == 0:
            raise ValueError("Weights add to zero, or ticker names don't match")
        elif self.weights is not None:
            new_weights = np.asarray(self.weights)
        else:
            raise ValueError("Weights is None")

        sigma = np.sqrt(objective_functions.portfolio_variance(new_weights, self.cov_matrix))

        if expected_returns is not None:
            mu = objective_functions.portfolio_return(
                new_weights, expected_returns, negative=False
            )
        return mu, sigma

def minimize_risk(stocks, target_return: float):
    print('Minimization risk, target return ', target_return)
    # Минимальный риск при заданной доходности
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = EfficientFrontier(mu, sigma, weight_bounds=(0,1)) 
    #min_return = min(i for i in mu.values if i > 0) # min return >0, drop if < 0?
    #max_risk = max(sigma.values)
    #print(max_risk)
    minrisk=ef.efficient_return(target_return)
    return ef

def maximize_return(stocks, target_risk: float):
    print('Max risk, target risk ', target_risk)
    # Максимальная доходность для заданного риска
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = EfficientFrontier(mu, sigma, weight_bounds=(0,1)) 
    maxret=ef.efficient_risk(target_risk)
    return ef

#port = get_min_risk(start='2018-01-01', end='2020-12-31', target_return=0.1)
#pwt=port.clean_weights()
#print("Weights", pwt)
#print("Portfolio performance:")
#print(port.portfolio_performance(verbose=True,))