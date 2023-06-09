import streamlit as st
from postgres.storage import register, login, load_ticker_history, connect
from markowitz import minimize_risk, maximize_return, getMaxReturn, getMinRisk, getStocksData, getMinReturn, getMaxRisk, getPortfolioHistory
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.reader import *

DETAILED_PIE = False
st.set_page_config(page_icon=":money_with_wings:", page_title="EMALAK Invest", layout="centered", initial_sidebar_state="collapsed")
sns.set_theme()
sns.set_context("notebook", font_scale=0.85, rc={"lines.linewidth": 2.5})
plt.rcParams['figure.facecolor'] = 'none'
cfg = read_config('config.json')
conn = connect()

def prettify_weighs(df):
    return df.sort_values(axis=1, by='Доля в портфеле (%)', ascending=False)


def build_pie(df):
    
    if df.shape[1] > 7 and not DETAILED_PIE:
        tickers = list(df.columns.values)
        values = list(df.values.tolist())
        top_tickers = tickers[:6]
        top_values = values[0][:6]
        other_values = values[0][6:]
        
        top_tickers.append("Другие акции")
        top_values.append(sum(other_values))
        fig, ax = plt.subplots()
        ax.pie(top_values, autopct='%1.2f%%')
        ax.axis('equal')
        ax.legend(labels=top_tickers)
        return fig
    
    fig, ax = plt.subplots()
    ax.pie(df.iloc[0], autopct='%1.2f%%')
    ax.axis('equal')
    ax.legend(labels=df.columns.values)
    return fig

def LogInClicked(username, password):
    if login(username, password):
        st.session_state['loggedIn'] = True
        st.session_state['username'] = username
        st.session_state['password'] = password
        #print('loggedIn')
    else:
        #st.session_state['loggedIn'] = False
        st.error('Неверный пароль или имя пользователя')

def RegisterClicked(username, password):
    if register(username, password):
        st.session_state['loggedIn'] = True
        st.session_state['username'] = username
        st.session_state['password'] = password
        #print('loggedIn')
    else:
        #st.session_state['loggedIn'] = False
        st.error('Неизвестная ошибка, попробуйте другое имя пользователя')

def goToRegistration():
    st.session_state.status = 'register'

def show_login_page():
    if st.session_state['loggedIn'] == False and st.session_state.status == 'login':
        username = st.text_input(label="username", label_visibility='hidden', value="", placeholder="Имя пользователя")
        password = st.text_input(label="password", label_visibility='hidden', value="", type="password", placeholder="Пароль")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Войти", on_click=lambda: LogInClicked(username, password))
        with col2:
            st.button("Зарегистрироваться", on_click=lambda: goToRegistration())

def LogOutClicked():
    st.session_state['loggedIn'] = False

def show_main_page():
    st.title('EMALAK Invest')
    st.title('Здравствуйте, ' + st.session_state['username'])
    col1, col2 = st.columns([1,1,])
    with col1:
        fromDate = st.date_input('От', value=datetime.date(2018, 1, 1), min_value=datetime.date(2015, 1, 1))
    with col2:
        toDate = st.date_input('До', min_value=fromDate)
    if ('stocks' not in st.session_state or 'fromDate' not in st.session_state or 'toDate' not in st.session_state) or (st.session_state.fromDate != fromDate or st.session_state.toDate != toDate):
        st.session_state.fromDate = fromDate
        st.session_state.toDate = toDate
        st.session_state.stocks = getStocksData(start=fromDate, end=toDate)

        min_risk = float(getMinRisk(st.session_state.stocks)) * 100 + 0.005
        max_risk = float(getMaxRisk(st.session_state.stocks)) * 100
        st.session_state.max_risk = max_risk
        st.session_state.min_risk = min_risk
        st.session_state.target_risk = min_risk
        
        max_return = float(getMaxReturn(st.session_state.stocks)) * 100 - 0.005
        min_return = float(getMinReturn(st.session_state.stocks)) * 100 + 0.005
        st.session_state.min_return = min_return
        st.session_state.max_return = max_return
        st.session_state.target_return = max_return
    if 'max_risk' not in st.session_state or (st.session_state.fromDate != fromDate or st.session_state.toDate != toDate):
        min_risk = float(getMinRisk(st.session_state.stocks)) * 100 + 0.005
        max_risk = float(getMaxRisk(st.session_state.stocks)) * 100
        st.session_state.max_risk = max_risk
        st.session_state.min_risk = min_risk
    if 'min_return' not in st.session_state or (st.session_state.fromDate != fromDate or st.session_state.toDate != toDate):
        max_return = float(getMaxReturn(st.session_state.stocks)) * 100 - 0.005
        min_return = float(getMinReturn(st.session_state.stocks)) * 100 + 0.005
        st.session_state.min_return = min_return
        st.session_state.max_return = max_return
        st.session_state.target_return = max_return
    if 'target_risk' not in st.session_state or (st.session_state.fromDate != fromDate or st.session_state.toDate != toDate):
        st.session_state.target_risk = st.session_state.min_risk
    if 'target_return' not in st.session_state or (st.session_state.fromDate != fromDate or st.session_state.toDate != toDate):
        st.session_state.target_return = st.session_state.max_return
    if 'deposit' not in st.session_state:
        st.session_state.deposit = 10000
    
    st.session_state.deposit = st.number_input('Депозит', min_value=10000, value=st.session_state.deposit)
    profile = st.radio('Профиль', ['Максимизация доходности', 'Минимизация риска'], index = 0)
    if profile == 'Максимизация доходности':
            st.session_state.target_risk = st.slider('Максимальный риск',min_value=st.session_state.min_risk, max_value=st.session_state.max_risk, step=0.01, value=st.session_state.target_risk)  
    elif profile == 'Минимизация риска':
            st.session_state.target_return = st.slider('Минимальная доходность',min_value=st.session_state.min_return, max_value=st.session_state.max_return, step=0.01, value=st.session_state.target_return)

    clicked = st.button('Расчитать')
    if clicked:
            if profile == 'Максимизация доходности':
                st.write('Расчет максимизации доходности при заданном риске в {:10.2f}%...'.format(st.session_state.target_risk))
                port = maximize_return(stocks=st.session_state.stocks, target_risk=(st.session_state.target_risk / 100))
                perf = port.portfolio_performance()
                weights = port.clean_weights()
                weights_cpy = weights.copy()
                for key, value in weights_cpy.items():
                    if value <= 0:
                        del weights[key]
                del weights_cpy
                expected_annual_return = perf[0]
                annual_risk = perf[1]
                col1, col2 = st.columns(2)
                col1.metric("Доходность", "%.2f" % (expected_annual_return * 100) + "%")
                col2.metric("Риск", "%.2f" % (annual_risk * 100) + "%")
                
                prweights = pd.DataFrame(weights, columns=weights.keys(), index=["Доля в портфеле (%)"])
                prweights = prweights.select_dtypes(exclude=['object', 'datetime']) * 100
                
                prweights = prettify_weighs(prweights)
                st.table(prweights)
                
                st.pyplot(build_pie(prweights),transparent=True)
                
                st.title('Динамика портфеля')
                history = getPortfolioHistory(st.session_state.deposit, weights, st.session_state.stocks)
                
                st.line_chart(data=history, x='date', y='value')
                
            elif profile == 'Минимизация риска':
                st.write('Расчет минимизации риска при заданной доходности в {:10.2f}%...'.format(st.session_state.target_return))
                port = minimize_risk(stocks=st.session_state.stocks,    target_return=(st.session_state.target_return / 100))
                perf = port.portfolio_performance()
                weights = port.clean_weights()
                weights_cpy = weights.copy()
                for key, value in weights_cpy.items():
                    if value <= 0:
                        del weights[key]
                del weights_cpy
                expected_annual_return = perf[0]
                annual_risk = perf[1]
                col1, col2 = st.columns(2)
                col1.metric("Доходность", "%.2f" % (expected_annual_return * 100) + "%")
                col2.metric("Риск", "%.2f" % (annual_risk * 100) + "%")
                
                prweights = pd.DataFrame(weights, columns=weights.keys(), index=["Доля в портфеле (%)"])
                prweights = prweights.select_dtypes(exclude=['object', 'datetime']) * 100
                
                prweights = prettify_weighs(prweights)
                st.table(prweights)
                
                st.pyplot(build_pie(prweights),transparent=True)
                
                st.title('Динамика портфеля')
                history = getPortfolioHistory(st.session_state.deposit, weights, st.session_state.stocks)
                
                fig = plt.figure()
                sns.lineplot(data=history, x='date', y='value')
                st.pyplot(fig)

def show_register_page():
        if st.session_state['loggedIn'] == False and st.session_state.status == 'register':
            username = st.text_input(label="username", key="regusername", label_visibility='hidden', value="", placeholder="Имя пользователя")
            password = st.text_input(label="password", key="regpassword",label_visibility='hidden', value="", type="password", placeholder="Пароль")
            st.button("Зарегистрироваться", key="regbutton", on_click=lambda: RegisterClicked(username, password))


if 'loggedIn' not in st.session_state:
    st.session_state['loggedIn'] = False
    st.session_state.status = 'login'
    show_login_page()
else:
    if st.session_state['loggedIn']:
        show_main_page()
    else:
        if 'status' not in st.session_state:
            st.session_state.status = 'login'
            show_login_page()
        else:
            if st.session_state.status == 'login':
                show_login_page()
            elif st.session_state.status == 'register':
                show_register_page()
