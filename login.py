import streamlit as st
from user import login
from marcovica import minimize_risk, maximize_return, getMaxReturn, getMinRisk, getStocksData
import datetime

# hashed_passwords = stauth.Hasher(['abc', 'def']).generate()   
st.set_page_config(page_icon=":money_with_wings:", page_title="EMALAK Invest", layout="centered", initial_sidebar_state="collapsed")

headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logoutSection = st.container()

def LogInClicked(username, password):
    print(username, password)
    if login(username, password):
        st.session_state['loggedIn'] = True
        st.session_state['username'] = username
        st.session_state['password'] = password
        #print('loggedIn')
    else:
        #st.session_state['loggedIn'] = False
        st.error('Неверный пароль или имя пользователя')

def show_login_page():
    with loginSection:
        if st.session_state['loggedIn'] == False:
            username = st.text_input(label="username", label_visibility='hidden', value="", placeholder="Имя пользователя")
            password = st.text_input(label="password", label_visibility='hidden', value="", type="password", placeholder="Пароль")
            st.button("Войти", on_click=lambda: LogInClicked(username, password))

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

        min_risk = float(getMinRisk(st.session_state.stocks)) * 100 + 0.01
        st.session_state.max_risk = min_risk
        st.session_state.min_risk = min_risk
        
        max_return = float(getMaxReturn(st.session_state.stocks)) * 100 - 0.01
        st.session_state.min_return = max_return
        st.session_state.max_return = max_return
    if 'max_risk' not in st.session_state:
        min_risk = float(getMinRisk(st.session_state.stocks)) * 100 + 0.01
        st.session_state.max_risk = min_risk
        st.session_state.min_risk = min_risk
    if 'min_return' not in st.session_state:
        max_return = float(getMaxReturn(st.session_state.stocks)) * 100 - 0.01
        st.session_state.min_return = max_return
        st.session_state.max_return = max_return

    profile = st.radio('Профиль', ['Максимизация доходности', 'Оптимальный', 'Минимизация риска'], index = 1)
    if profile == 'Максимизация доходности':
            st.session_state.max_risk = st.slider('Максимальный риск',min_value=st.session_state.min_risk, max_value=30.0, step=0.01, value=st.session_state.max_risk)  
    elif profile == 'Минимизация риска':
            st.session_state.min_return = st.slider('Минимальная доходность',min_value=0.0, max_value=st.session_state.max_return, step=0.01, value=st.session_state.min_return)

    clicked = st.button('Расчитать')
    if clicked:
            if profile == 'Максимизация доходности':
                st.write('Расчет максимизации доходности при заданном риске в {:10.2f}%...'.format(st.session_state.max_risk))
                port = maximize_return(stocks=st.session_state.stocks, target_risk=(st.session_state.max_risk / 100))
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
                st.write(weights)
            elif profile == 'Оптимальный':
                st.write('Расчет оптимального портфеля')
            elif profile == 'Минимизация риска':
                st.write('Расчет минимизации риска при заданной доходности в {:10.2f}%...'.format(st.session_state.min_return))
                port = minimize_risk(stocks=st.session_state.stocks,    target_return=(st.session_state.min_return / 100))
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
                st.write(weights)



    
with headerSection:
    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = False
        show_login_page()
    else:
        if st.session_state['loggedIn']:
            show_main_page()
        else:
            show_login_page()
