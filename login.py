import streamlit as st
import streamlit_authenticator as stauth
from user import login

# hashed_passwords = stauth.Hasher(['abc', 'def']).generate()   
st.set_page_config(page_icon=":money_with_wings:", page_title="Акции", layout="wide", initial_sidebar_state="collapsed")

headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logoutSection = st.container()

def LogInClicked(username, password):
    print(username, password)
    if login(username, password):
        st.session_state['loggedIn'] = True
        print('loggedIn')
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
    st.title('Main page')
    st.button("logout", on_click=lambda: LogOutClicked())

    
with headerSection:
    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = False
        show_login_page()
    else:
        if st.session_state['loggedIn']:
            show_main_page()
        else:
            show_login_page()
