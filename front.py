import streamlit as st
import pandas as pd
import numpy as np
import requests
import parsing.finviz.finviz as finviz
import parsing.yahoo.yahoo as yahoo

from streamlit.components.v1 import html

def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date()     - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)

st.set_page_config(page_icon=":money_with_wings:", page_title="Акции", layout="wide", initial_sidebar_state="collapsed")

st.title('Анализ финансовых данных компании')
if st.button("APPL"):
    nav_page("details")

ticker = st.text_input("Ticker", "")
if ticker:
    st.write('The current ticker is', ticker)
    #загрузим общую информацию
    data_load_state = st.text('Загружаем базовую информацию...')
    df = yahoo.parse_general_info(ticker)
    data_load_state.text('Информация... Загружена!')

    st.subheader('Общая информация')
    st.table(pd.read_json(df.to_json()))

    history_load_state = st.text('Загружаем историческую информацию...')
    history = yahoo.get_historical_data(ticker)
    history_load_state.text('Информация... Загружена!')

    st.subheader('Историческая информация')
    st.text('Изменение биржевых котировок:')
    st.line_chart(history['Adj Close**'])

    st.text('Изменение объема торгов:')
    st.line_chart(history['Volume'])

    ma_day = [10,20,30]

    for ma in ma_day:
        column_name = "MA for %s days" %(str(ma))
        history[column_name] = history['Adj Close**'].rolling(window=ma,center=False).mean()

    st.text('Скользящие средние:')
    st.line_chart(history[['Adj Close**','MA for 10 days','MA for 20 days','MA for 30 days']])
