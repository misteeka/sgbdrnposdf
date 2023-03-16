import yfinance as yf
import numpy as np
import pandas as pd
from optimizer import Optimizer
from postgres.storage import load_ticker_history

# Получение данных по ценам акций
def getStocksData(start, end):
    #tickers = ['LKOH.ME','GMKN.ME', 'DSKY.ME', 'NKNC.ME', 'MTSS.ME', 'IRAO.ME', 'SBER.ME', 'AFLT.ME']
    #df_stocks= yf.download(tickers, start=start, end=end)['Adj Close']
    #df_stocks.head()
    #nullin_df = pd.DataFrame(df_stocks,columns=tickers)
    #nullin_df.isnull().sum()
    return load_ticker_history()

# Получение минимального дохода
def getMinReturn(stocks):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = Optimizer(mu, sigma, weight_bounds=(0,1)) 
    ef.efficient_return(float(0))
    return ef.portfolio_performance()[0]

# Получение максимально возможного риска портфеля
def getMaxRisk(stocks):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = Optimizer(mu, sigma, weight_bounds=(0,1)) 
    ef.efficient_risk(float(1))
    return ef.portfolio_performance()[1]

# Годовая доходность
def get_mu(prices):
    frequency = 252 # TODO
    if not isinstance(prices, pd.DataFrame):
        print("prices are not in a dataframe")
        prices = pd.DataFrame(prices)
    returns = prices.pct_change().dropna(how="all")
    return (1 + returns).prod() ** (frequency / returns.count()) - 1

def _is_positive_semidefinite(matrix):
    try:
        # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False
    
def fix_nonpositive_semidefinite(matrix):
    print('a')
    if _is_positive_semidefinite(matrix):
        return matrix
    print('b')

    # Eigendecomposition
    q, V = np.linalg.eigh(matrix)

    # Remove negative eigenvalues
    q = np.where(q > 0, q, 0)
    # Reconstruct matrix
    fixed_matrix = V @ np.diag(q) @ V.T
    
    if not _is_positive_semidefinite(fixed_matrix):  # pragma: no cover
        print("Could not fix matrix.")

    # Rebuild labels if provided
    if isinstance(matrix, pd.DataFrame):
        tickers = matrix.index
        return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
    else:
        return fixed_matrix

# Cov
def get_sigma(prices):
    frequency = 252 # TODO
    if not isinstance(prices, pd.DataFrame):
        print("data is not in a dataframe")
        prices = pd.DataFrame(prices)
    returns = prices.pct_change().dropna(how="all")
    return fix_nonpositive_semidefinite(returns.cov() * frequency)

# Получение минимально возможного риска портфеля
def getMinRisk(stocks):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = Optimizer(mu, sigma, weight_bounds=(0,1)) 
    ef.min_volatility()
    return ef.portfolio_performance()[1]

# Получение максимально возможного риска портфеля
def getMaxReturn(stocks):
    return max(get_mu(stocks).values)

# Минимальный риск при заданной доходности
def minimize_risk(stocks, target_return: float):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = Optimizer(mu, sigma, weight_bounds=(0,1))
    minrisk=ef.efficient_return(target_return)
    return ef

# Максимальная доходность для заданного риска
def maximize_return(stocks, target_risk: float):
    mu = get_mu(stocks)
    sigma = get_sigma(stocks)
    ef = Optimizer(mu, sigma, weight_bounds=(0,1)) 
    maxret=ef.efficient_risk(target_risk)
    return ef

#port = minimize_risk(getStocksData(start='2018-01-01', end='2020-12-31'), target_return=0.25)
#pwt=port.clean_weights()
#print("Weights", pwt)
#print("Portfolio performance:")
#print(port.portfolio_performance())