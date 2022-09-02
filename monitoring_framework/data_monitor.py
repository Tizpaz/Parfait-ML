from unicodedata import category
from streamlit_plotly_events import plotly_events
import streamlit as st
import numpy as np
import plotly.express as plt
import plotly.tools as tools
import math
import matplotlib.pyplot as mplt


import sys

sys.path.append("../")


def main():
    from subjects.adf_data.census import census_data
    from subjects.adf_data.credit import credit_data
    from subjects.adf_data.bank import bank_data
    from subjects.adf_data.compas import compas_data
    import pandas as pd
    vis_technique = ["Correlation", "PCA", "LDA"]
    datasets = [("census", "gender",9), ("census", "race",8), ("credit", "gender",9), ("bank","age",1), ("compas","gender",1), ("compas","race",3)]

    picked_vis_technique = st.radio("Visualization technique:", vis_technique, horizontal = True)
    picked_dataset = st.radio("Dataset:", datasets, horizontal = True)
    sensitive_param = picked_dataset[2]
    sensitive_name = picked_dataset[1]
    
    if picked_vis_technique == "Correlation":
        from sklearn.covariance import EmpiricalCovariance
        from sklearn.preprocessing import StandardScaler
        data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}
        X, Y, input_shape, nb_classes = data[picked_dataset[0]]()
        X_transformed = StandardScaler().fit_transform(X) # This turns the "covariance" into correlation
        Y = np.argmax(Y, axis=1)
        cov = EmpiricalCovariance().fit(X_transformed)
        # bar_data = {"Column number": nb_classes, "Correlation": cov.covariance_[sensitive_param-1]}
        sensitive_feature = [False]*len(cov.covariance_[sensitive_param-1])
        sensitive_feature[sensitive_param-1] = True
        bar_data = {"Column number": range(len(cov.covariance_[sensitive_param-1])), "Correlation": cov.covariance_[sensitive_param-1], "Sensitive feature": sensitive_feature}
        bar_data = pd.DataFrame(bar_data)
        fig = plt.bar(bar_data, x="Column number", y="Correlation", color="Sensitive feature",category_orders={"Sensitive feature": [False, True]}, color_discrete_sequence=['blue', 'red'], title=f"Correlation of each column with {sensitive_name}")
        bar_graph = plotly_events(fig)

