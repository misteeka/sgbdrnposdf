import yfinance as yf
import numpy as np
import pandas as pd
from optimizer import Optimizer

# Получение данных по ценам акций
def getStocksData(start, end):
    tickers = ['LKOH.ME','GMKN.ME', 'DSKY.ME', 'NKNC.ME', 'MTSS.ME', 'IRAO.ME', 'SBER.ME', 'AFLT.ME']
    
    df_stocks= yf.download(tickers, start='2018-01-01', end='2021-01-01')['Adj Close']
    #df_stocks.head()
    nullin_df = pd.DataFrame(df_stocks,columns=tickers)
    nullin_df.dropna()
    return df_stocks

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
    frequency = 251 # TODO
    returns = prices.pct_change().dropna(how="all")
    #print('SGBDRN IDRN ', frequency / returns.count())
    return (1 + returns).prod() ** (frequency / returns.count()) - 1

def _is_positive_semidefinite(matrix):
    try:
        # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False
    
def fix_nonpositive_semidefinite(matrix):
    if _is_positive_semidefinite(matrix):
        return matrix
    print('WTF')

    # Eigendecomposition
    q, V = np.linalg.eigh(matrix)

    # Remove negative eigenvalues
    q = np.where(q > 0, q, 0)
    # Reconstruct matrix
    fixed_matrix = V @ np.diag(q) @ V.T

    # Rebuild labels if provided
    if isinstance(matrix, pd.DataFrame):
        tickers = matrix.index
        return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
    else:
        return fixed_matrix

# Cov
def get_sigma(prices):
    frequency = 252 # TODO
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

def getPortfolioHistory(deposit, weights, stocks):
    #print(deposit)
    #print(weights)
    amounts = dict()
    for w in weights:
        price = stocks[w][0]
        amount = (deposit * weights[w]) / price
        amounts[w] = amount
    #print(stocks)
    #print(amounts)
    value = []
    dates = []

    for date, s in stocks.iterrows():
        cost = 0
        for a in amounts:
            cost += s[a] * amounts[a]
        
        dates.append(date)
        value.append(cost)

    perf = pd.DataFrame({'value': value, 'date': dates})
    #print(perf)
    return perf
#df = getStocksData(start='2018-01-01', end='2021-03-15')
#port = minimize_risk(df, target_return=0.25)
#pwt=port.clean_weights()
#print("Weights", pwt)
#print("Portfolio performance:")
#print(port.portfolio_performance())
#getAmounts(10000, pwt, df)