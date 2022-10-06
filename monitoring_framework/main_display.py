import data_monitor
import learned_model_monitor
import home
import streamlit as st

version = st.radio("Visualization tool:", ["Home", "Data Visualization", "Learned Model Visualization"], horizontal = True)

if version == "Home":
    home.main()
if version == "Data Visualization":
    data_monitor.main()
elif version == "Learned Model Visualization":
    learned_model_monitor.main()
