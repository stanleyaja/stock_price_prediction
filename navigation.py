import streamlit as st
from streamlit_option_menu import option_menu

from predict import view_predict
from about import view_about

selected = option_menu(
    menu_title="Stock Price Prediction",
    options=["Predict", "About"],
    icons=["graph-up arrow", "info-circle"],
    menu_icon="clipboard-data",
    default_index=0,
    orientation="horizontal"
)

def handleSelectedPage(selected):
    if selected == "Predict":
        view_predict()
    elif selected == "About":
        view_about()

handleSelectedPage(selected)