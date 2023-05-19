import numpy as np
import pandas as pd
import plotly.express as plt
import plotly.tools as tools
import math
import pickle
from Themis.Themis2.themis2 import Themis
import os
import matplotlib.pyplot as mplt
from configs import columns, get_groups, labeled_df, categorical_features, categorical_features_names, int_to_cat_labels_map, cat_to_int_map

import sys

sys.path.append("./")
sys.path.append("../")



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

# Plots all the graphs
def create_model(model, dataset, algo):

    # Plot the pareto optimal frontier

    df = pd.read_csv(os.path.dirname(__file__)+"/../Dataset" + "/" + f"{model}_{dataset[0]}_{dataset[1]}_{algo}.csv")
    df_masking = df.copy()
    df_masking["score"] = -df_masking["score"] # we want to find maximium for score
    mask = is_pareto_efficient(df_masking[["score","AOD"]].to_numpy(), True)
    df = df.assign(pareto_optimal=mask)

    # Get themis data
    df = df.assign(themis_group_score="NA")
    df = df.assign(themis_causal_score="NA")
    df_pareto_optimal = df[df["pareto_optimal"]]
    count = 0

    df_worst = df[df["AOD"] == df["AOD"].max()]
    df_pareto_optimal_max_AOD_row = df_pareto_optimal[df_pareto_optimal["AOD"] == df_pareto_optimal["AOD"].min()]
    df_pareto_optimal_max_score_row = df_pareto_optimal[df_pareto_optimal["score"] == df_pareto_optimal["score"].max()]

    for row in [df_worst, df_pareto_optimal_max_AOD_row, df_pareto_optimal_max_score_row]:

        themis_studying_feature = [dataset[1] if dataset[1] != "gender" else "sex"]
        tests = [{"function": "causal_discrimination", "threshold": 0.2, "conf": 0.98, "margin": 0.02, "input_name": themis_studying_feature},
            {"function": "group_discrimination", "threshold": 0.2, "conf": 0.98, "margin": 0.02, "input_name": themis_studying_feature}]
        write_file = row['write_file'].iloc[0]
        file_path = os.path.realpath(os.path.dirname(__file__))
        file = open(file_path+"/"+f".{write_file}", "rb")
        trained_model = pickle.load(file, encoding="latin-1")
        S = Themis_S(trained_model, dataset)
        themis_results = Themis(S, tests, f"{file_path}/Themis/Themis2/settings_{dataset[0]}.xml").run()
        causal_answer = themis_results[0][1][1]
        group_answer = themis_results[1][1][1]
        row["themis_group_score"] = group_answer
        row["themis_causal_score"] = causal_answer
        count += 1
    

    return df_worst[["score", "AOD", "themis_group_score", "themis_causal_score"]], df_pareto_optimal_max_score_row[["score", "AOD", "themis_group_score", "themis_causal_score"]], df_pareto_optimal_max_AOD_row[["score", "AOD", "themis_group_score", "themis_causal_score"]]

    


def Themis_S(trained_model, dataset):
    def Themis_S(x):
        intergerized_x = []
        for i in range(len(x)):
            if i in categorical_features[dataset[0]][:-1]:
                intergerized_x.append(int(cat_to_int_map(dataset[0])[columns[dataset[0]][i]][x[i]]))
            else:
                intergerized_x.append(int(x[i]))
        return list(trained_model.predict([intergerized_x])) == [trained_model.classes_[0]]
    return Themis_S


def convert_df(df):
   return df.to_csv("tested_themis_summary.csv", index=False)

def main():
    models = ["LR","RF","SV","DT"]
    models = ["Logistic Regression","Random Forest","Support Vector Machine","Decision Tree"]

    models_key = {"Logistic Regression":"LR", "Random Forest": "RF", "Support Vector Machine": "SV", "Decision Tree": "DT"}
    # Note: Index starts at 1 for the datasets, so subtract 1 from datasets[2] to ensure we are highlighting the correct sensitive feature!
    datasets = [("census", "gender",9), ("census", "race",8), ("credit", "gender",9), ("bank","age",1), ("compas","gender",1), ("compas","race",3)]

    algorithms = ["mutation"]

    # Basic buttons to choose the correct models
    
    picked_algo = "mutation"
    count = 0
    score_dataframe = pd.DataFrame(columns = ['score', 'AOD', 'themis_group_score', 'themis_causal_score', 'dataset', 'model', 'optimal'])

    for d in datasets:
        for m in models:
            model_scores = create_model(models_key[m], d, picked_algo)

            model_score_worst = model_scores[0]
            model_score_worst['dataset'] = d[0] + ", " + d[1]
            model_score_worst['model'] = m
            model_score_worst['optimal'] = "Worst"

            model_score_acc = model_scores[1]
            model_score_acc['dataset'] = d[0] + ", " + d[1]
            model_score_acc['model'] = m
            model_score_acc['optimal'] = "Score"

            model_score_AOD = model_scores[2]
            model_score_AOD['dataset'] = d[0] + ", " + d[1]
            model_score_AOD['model'] = m
            model_score_AOD['optimal'] = "Fairness"
            score_dataframe = score_dataframe.append(model_score_worst.iloc[0], ignore_index=True)
            score_dataframe = score_dataframe.append(model_score_acc.iloc[0], ignore_index=True)
            score_dataframe = score_dataframe.append(model_score_AOD.iloc[0], ignore_index=True)
            count +=1

    convert_df(score_dataframe)

    
if __name__ == "__main__":
    main()