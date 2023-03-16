import pandas as pd
import apimoex
import requests
from datetime import date
from json import load
from postgres.storage import *

def get_historical(ticker):
    with requests.Session() as session:
        data = apimoex.get_board_history(session, ticker)
    df = pd.DataFrame(data)
    df = df.drop('BOARDID', axis=1)
    df = df.dropna()
    df['unix_time'] = pd.to_datetime(df.TRADEDATE).astype(int) / 10**9
    return df
    

def cut_historical(df, start_date, end_date):
    mask = (df['TRADEDATE'] > start_date) & (df['TRADEDATE'] <= end_date)
    return df.loc[mask]

def load_historical(tickers, start_date, end_date):
    conn = connect()
    out = pd.DataFrame()
    frames = []
    for ticker in tickers:
        df = load_ticker_history(conn, ticker, start_date, end_date)
        frames.append(df)
    out['date'] = frames[0]['TRADEDATE']
    
    for ticker, df in zip(tickers, frames):
        out[ticker] = df['CLOSE']
    out = out.fillna(out.mean())
    out = out.set_index('date')
    return out