from streamlit_plotly_events import plotly_events
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as plt

models = ["LR","RF","SV","DT"]
datasets = [("census", "gender"), ("census", "race"), ("credit", "gender"), ("bank","age"), ("compas","gender"), ("compas","race")]

algorithms = ["mutation", "masking"]

picked_model = st.radio("Model:", models, horizontal = True)
picked_dataset = st.radio("Dataset:", datasets, horizontal = True)
picked_algo = st.radio("Dataset:", algorithms, horizontal = True)


def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def create_model(model, dataset, algo):
    df = pd.read_csv("../Dataset" + "/" + f"{model}_{dataset[0]}_{dataset[1]}_{algo}.csv")
    df_masking = df.copy()
    df_masking["score"] = -df_masking["score"] # we want to find maximium for score
    mask = is_pareto_efficient(df_masking[["score","AOD"]].to_numpy(), True)
    df = df.assign(pareto_optimal=mask)
    fig = plt.scatter(df, x = "score", y = "AOD", color='pareto_optimal')
    # plt.scatter(df["score"][mask],df["AOD"][mask], color='red',s=5)
    # plt.axis((.5,1,0,.25))
    # plt.title(f"Pareto optimal frontier pointsfor {model}_{dataset[0]}_{dataset[1]}_{algo}")
    selected_points = plotly_events(fig)
    if selected_points is not None and len(selected_points) > 0:
        selected_point = selected_points[0]['pointIndex']
        st.write(f"Hyperparameters: {df.iloc[selected_point]['inp']}")


create_model(picked_model, picked_dataset, picked_algo)