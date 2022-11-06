import data_monitor
import learned_model_monitor
import study_counterfactuals
import home
import streamlit as st
st.write("NOTE: clicking on some of the plotly graphs produces more details.")
version = st.radio("Visualization tool:", ["Home", "Data Visualization", "Learned Model Explainations", "Study counterfactuals"], horizontal = True)

if version == "Home":
    home.main()
if version == "Data Visualization":
    data_monitor.main()
elif version == "Learned Model Explainations":
    learned_model_monitor.main()
elif version == "Study counterfactuals":
    study_counterfactuals.main()