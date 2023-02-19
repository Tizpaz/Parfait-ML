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

from subjects.adf_data.census import census_data
from subjects.adf_data.credit import credit_data
from subjects.adf_data.bank import bank_data
from subjects.adf_data.compas import compas_data
data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}



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


def num_parfait_counterfactuals(trained_model, dataset):

    X, Y, input_shape, nb_classes = data[dataset[0]]()
    Y = np.argmax(Y, axis=1)
    X = X.astype(np.int64)
    Y = Y.astype(np.int64)
    nonlabeled_data = pd.concat([pd.DataFrame(X),pd.DataFrame(Y)], axis=1)
    nonlabeled_data.columns = columns[dataset[0]]
    labeled_data = labeled_df(pd.concat([pd.DataFrame(X),pd.DataFrame(Y)], axis=1), dataset[0])
    labeled_data.columns = columns[dataset[0]]

    # Get predicted probability for each data point
    predictions_probs = trained_model.predict_proba(X)
    # (predictions_probs)
    prediction_probabilities = pd.DataFrame(np.array(predictions_probs)[:,0], columns=["Probability"])
    predictions = trained_model.predict(X)
    prediction_probabilities["label"] = predictions
    prediction_probabilities["correctness"] = (predictions == Y)
    prediction_probabilities["label"] = prediction_probabilities["label"].astype("string").map(int_to_cat_labels_map(dataset[0])['label'])
    prediction_probabilities["correctness"] = prediction_probabilities["correctness"].map({False: "Incorrect", True: "Correct"})
    # prediction_probabilities["label_correctness"] = prediction_probabilities["label"] + ", "+prediction_probabilities["prediction_correct"]

    studying_feature = columns[dataset[0]][:-1][dataset[2]-1]

    feature_index = columns[dataset[0]][:-1].index(studying_feature)
    prediction_probabilities[f"counter_factual_{studying_feature}"] = False

    all_predictions = []
    if feature_index in categorical_features[dataset[0]][:-1]:
        for possible_sensitive_value in list(int_to_cat_labels_map(dataset[0])[studying_feature].keys()):
            counter_factuals_X = X.copy()
            
            counter_factuals_X[:,feature_index] = possible_sensitive_value
            counter_factual_predictions = trained_model.predict(counter_factuals_X)
            all_predictions.append(counter_factual_predictions)
            prediction_probabilities[f"counter_factual_{studying_feature}"] = prediction_probabilities[f"counter_factual_{studying_feature}"] | (counter_factual_predictions != predictions)
    else:
        counter_factuals_X = X.copy()
        counter_factuals_X[:,feature_index] = X[:,feature_index].max()
        counter_factual_max_predictions = trained_model.predict(counter_factuals_X)
        counter_factuals_X[:,feature_index] = X[:,feature_index].min()
        counter_factual_min_predictions = trained_model.predict(counter_factuals_X)
        prediction_probabilities[f"counter_factual_{studying_feature}"] = prediction_probabilities[f"counter_factual_{studying_feature}"] | (counter_factual_max_predictions != predictions) | (counter_factual_min_predictions != predictions)
        
    num_counter_factuals = len(prediction_probabilities[prediction_probabilities[f'counter_factual_{studying_feature}'] == True])
    num_incorrect_counter_factuals = len(prediction_probabilities[(prediction_probabilities[f'counter_factual_{studying_feature}'] == True) & (prediction_probabilities['correctness'] == 'Incorrect')])
    num_correct_counter_factuals = len(prediction_probabilities[(prediction_probabilities[f'counter_factual_{studying_feature}'] == True) & (prediction_probabilities['correctness'] == 'Correct')])
    return num_counter_factuals, len(prediction_probabilities)

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
    # count = 0

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
        row["num_parfait_counterfactuals"], row["total_sampled_points"] = num_parfait_counterfactuals(trained_model, dataset)
    

    return df_worst[["num_parfait_counterfactuals", "total_sampled_points"]], df_pareto_optimal_max_score_row[["num_parfait_counterfactuals", "total_sampled_points"]], df_pareto_optimal_max_AOD_row[["num_parfait_counterfactuals", "total_sampled_points"]]

    


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
   return df.to_csv("tested_parfait-ml_counterfactual_summary.csv", index=False)

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
    score_dataframe = pd.DataFrame(columns = ["num_parfait_counterfactuals","total_sampled_points", 'dataset', 'model', 'optimal'])

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