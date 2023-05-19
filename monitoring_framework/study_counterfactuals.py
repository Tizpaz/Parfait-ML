from streamlit_plotly_events import plotly_events
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as plt
import plotly.tools as tools
import math
import random
import os
from scipy.optimize import dual_annealing
import matplotlib.pyplot as mplt
import re
from configs import columns, get_groups, labeled_df, categorical_features, categorical_features_names, int_to_cat_labels_map, cat_to_int_map
from subjects.adf_data.census import census_data
from subjects.adf_data.credit import credit_data
from subjects.adf_data.bank import bank_data
from subjects.adf_data.compas import compas_data
data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}
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
    zoom = st.slider("Zoom: ", min_value=0.0, max_value=1.0, step=.0001)
    st.write("Each dot below represents a model. Click on a dot to gain further insight on model parameters/decisions and classification explainability!")
    fig = plt.scatter(df, x = "score", y = "AOD", color='pareto_optimal', category_orders={'pareto_optimal': [False, True]}, color_discrete_sequence=['blue', 'red'],
            title="Pareto Optimal Frontier", range_x=[zoom*.5+.5,1], range_y=[-0.01,.25-zoom*.25])
    

    # "Listener" event for when the pareto optimal frontier is clicked on, determines the point that is clicked
    selected_points = plotly_events(fig)
    if selected_points is not None and len(selected_points) > 0:
        import pickle
        
        selected_point = selected_points[0]['pointIndex']
        write_file = df.iloc[selected_point]['write_file']
        st.write(f"Hyperparameters: {write_file}")
        file = open(os.path.dirname(__file__)+"/"+f".{write_file}", "rb")
        trained_model = pickle.load(file, encoding="latin-1")
        
        if model == "LR": # For logistic regression, a bar graph is created to show the weights of the model
            X, Y, input_shape, nb_classes = data[dataset[0]]()
            model_df = {"feature": columns[dataset[0]][:-1], "weight":trained_model.coef_[0], "std_adj_weights":trained_model.coef_[0]*np.std(X, axis=0), "sensitive_feature": [False]*len(trained_model.coef_[0])}
            model_df["sensitive_feature"][dataset[2]-1] = True
            fig_2 = plt.bar(model_df, x="feature", y="weight", color="sensitive_feature", category_orders={"feature": columns[dataset[0]][:-1]},color_discrete_map={False:'blue', True:'red'},
                title="The LR Model's coefficient/weights")
            fig_3 = plt.bar(model_df, x="feature", y="std_adj_weights", color="sensitive_feature", category_orders={"feature": columns[dataset[0]][:-1]},color_discrete_map={False:'blue', True:'red'},
                title="The LR Model's standard deviation adjusted weights")
            plotly_events(fig_2)
            plotly_events(fig_3)
        elif model == "RF": # For random forest, trees are graphed to show the logic of the model
            from sklearn import tree
            import graphviz as graphviz
            
            checked = []
            rows = []
            n_cols = 10
            st.write("Display tree:")
            for i in range(len(trained_model.estimators_)):
                if i%n_cols == 0:
                    rows.append(st.columns(n_cols))
                checked.append(rows[math.floor(i/n_cols)][i%n_cols].checkbox(label=f"{i}"))
            
            max_depth = st.slider('Max depth', 0, 10, 2)
            
            feature_names = []
            for i in range(trained_model.n_features_in_):
                if not i == dataset[2]-1:
                    feature_names.append(f"feature {i}")
                else:
                    feature_names.append(dataset[1])


            for i in range(len(checked)):
                if checked[i]:
                    dot_data = tree.export_graphviz(trained_model.estimators_[i],max_depth=max_depth, feature_names=columns[dataset[0]][:-1], proportion = True, class_names=True, out_file=None, rounded=True)
                    st.graphviz_chart(dot_data)
        elif model == "SV":
            X, Y, input_shape, nb_classes = data[dataset[0]]()
            model_df = {"feature": columns[dataset[0]][:-1], "weight":trained_model.coef_[0], "std_adj_weights":trained_model.coef_[0]*np.std(X, axis=0), "sensitive_feature": [False]*len(trained_model.coef_[0])}
            model_df["sensitive_feature"][dataset[2]-1] = True
            fig_2 = plt.bar(model_df, x="feature", y="weight", color="sensitive_feature", category_orders={"feature": columns[dataset[0]][:-1]}, color_discrete_map={False:'blue', True:'red'},
                title="The SVM Model's hyperplane slopes")
            fig_3 = plt.bar(model_df, x="feature", y="std_adj_weights", color="sensitive_feature", category_orders={"feature": columns[dataset[0]][:-1]},color_discrete_map={False:'blue', True:'red'},
                title="The SVM Model's standard deviation adjusted hyperplane slopes")
            plotly_events(fig_2)
            plotly_events(fig_3)

        elif model == 'DT': # Just show the one model
            from sklearn import tree
            import graphviz as graphviz

            feature_names = []
            for i in range(trained_model.n_features_in_):
                if not i == dataset[2]-1:
                    feature_names.append(f"feature {i}")
                else:
                    feature_names.append(dataset[1])
            max_depth = st.slider('Max depth', 0, 10, 2)
            dot_data = tree.export_graphviz(trained_model,max_depth=max_depth, feature_names=columns[dataset[0]][:-1], proportion = True, class_names=True, out_file=None, rounded=True)
            st.graphviz_chart(dot_data)
        

        # We now seek to explain a datapoint and how it is predicted
        st.write("Counter-factuals study")
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
        print(predictions_probs)
        prediction_probabilities = pd.DataFrame(np.array(predictions_probs)[:,0], columns=["Probability"])
        predictions = trained_model.predict(X)
        prediction_probabilities["label"] = predictions
        prediction_probabilities["correctness"] = (predictions == Y)
        prediction_probabilities["label"] = prediction_probabilities["label"].astype("string").map(int_to_cat_labels_map(dataset[0])['label'])
        prediction_probabilities["correctness"] = prediction_probabilities["correctness"].map({False: "Incorrect", True: "Correct"})
        # prediction_probabilities["label_correctness"] = prediction_probabilities["label"] + ", "+prediction_probabilities["prediction_correct"]


        color_on = st.radio("Color on:", options=["correctness", "label"], horizontal=True)
        prob_fig = plt.scatter(prediction_probabilities, x=prediction_probabilities.index, y="Probability", title="Prediction probability for each data point in dataset",
            color=color_on, symbol="label" if color_on == "correctness" else "correctness", labels={"Probability": f"Probability of {list(int_to_cat_labels_map(dataset[0])['label'].values())[0]}"})
        selected_explain_points = plotly_events(prob_fig)

        

        st.write("Counter factuals at a glance")
        studying_feature = st.selectbox("Select feature to study", columns[dataset[0]][:-1], index=dataset[2]-1)

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
            

        st.write("Categorical features will be studied by trying out all possible catories. Numerical features will be studied by trying out the maximium and minimium number in the training dataset.")

        num_counter_factuals = len(prediction_probabilities[prediction_probabilities[f'counter_factual_{studying_feature}'] == True])
        num_incorrect_counter_factuals = len(prediction_probabilities[(prediction_probabilities[f'counter_factual_{studying_feature}'] == True) & (prediction_probabilities['correctness'] == 'Incorrect')])
        num_correct_counter_factuals = len(prediction_probabilities[(prediction_probabilities[f'counter_factual_{studying_feature}'] == True) & (prediction_probabilities['correctness'] == 'Correct')])
        
        num_correct = len(prediction_probabilities[prediction_probabilities['correctness'] == 'Correct'])
        num_incorrect = len(prediction_probabilities[prediction_probabilities['correctness'] == 'Incorrect'])
        st.write(f"There are {num_counter_factuals} counter-factuals out of {len(prediction_probabilities)} data-points.")
        st.write(f"There are {num_incorrect_counter_factuals} currently incorrectly labeled counter-factuals out of {num_incorrect} incorrect data-points.")
        st.write(f"There are {num_correct_counter_factuals} currently correctly labeled counter-factuals out of {num_correct} correct data-points.")
        st.write("Click on a data-point below to test it out specifically")
        prob_fig_counterfactuals = plt.scatter(prediction_probabilities, x=prediction_probabilities.index, y="Probability", title=f"Counter-factual datapoints on {studying_feature}",
            color=f'counter_factual_{studying_feature}', color_discrete_map={False: "blue", True: "red"}, symbol="correctness", labels={"Probability": f"Probability of {list(int_to_cat_labels_map(dataset[0])['label'].values())[0]}"})
        selected_explain_points_counterfactuals = plotly_events(prob_fig_counterfactuals)

        import sklearn
        import lime
        import lime.lime_tabular
        from io import BytesIO
        from PIL import Image
        train, test, labeled_train, labeled_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.80)
        categorical_names = {}
        # Processing features so that it fits for LIME

        labeled_X_np = labeled_data.to_numpy()[:,:-1] # to array, minus the label
        print(labeled_X_np)
        feature_label_encoders = {}
        for feature in categorical_features[dataset[0]][:-1]:
            le_x = sklearn.preprocessing.LabelEncoder()
            le_x.fit(labeled_X_np[:, feature])
            feature_label_encoders[feature] = le_x
            labeled_X_np[:, feature] = le_x.transform(labeled_X_np[:, feature])
            categorical_names[feature] = list(le_x.classes_)
        
        
        labeled_Y_np = labeled_data.to_numpy()[:,-1:] # to array, minus the label
        le_y = sklearn.preprocessing.LabelEncoder()
        le_y.fit(labeled_Y_np)
        labeled_Y_np = le_y.transform(labeled_Y_np)
        class_names = le_y.classes_

        if (selected_explain_points_counterfactuals is None or len(selected_explain_points_counterfactuals) == 0):
            
            # num_points = st.number_input("Number of points to sample", min_value=0, value=min(50,num_counter_factuals) )

            st.write("Randomly sampled points LIME weighting")
            

            # Use simulated annealing
            NUM_MCMC_SAMPLES = 3

            # TO ADAPT TO HOME
            feature_sensitive_percentage = {}
            percentages_averages = {}
            explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = columns[dataset[0]][:-1],class_names=list(class_names),
                                                    categorical_features=categorical_features[dataset[0]][:-1], 
                                                    categorical_names=categorical_names, random_state=1, kernel_width=3)
            predict_fn = lambda x: trained_model.predict_proba(x)
            category = "sex" if dataset[1] == "gender" else dataset[1]
            group_0, group_1 = get_groups(dataset[0], category, get_name=True)
            for col in columns[dataset[0]]:
                
                feature_sensitive_percentage[col] = labeled_data[col][(labeled_data[category]==group_0)].value_counts()/(labeled_data[col].value_counts())*100
                percentages_averages[col] = labeled_data[labeled_data[category]==group_0][category].count()/labeled_data[category].count()*100
            
            def sensitive_discrimination_score(samples, reason = False):
                assert(len(samples) == NUM_MCMC_SAMPLES)
                feature_discrimination_impact = {col:0 for col in columns[dataset[0]][:-1]}
                for data_point in samples:
                    data_point = int(data_point)
                    if reason:
                        num_samples=50000
                    else:
                        num_samples = 5000
                    exp_reg = explainer.explain_instance(labeled_X_np[data_point], predict_fn, num_features=len(columns[dataset[0]][:-1]), num_samples=num_samples)
                    for feature_entry in exp_reg.as_map()[1]:
                        cur_column = columns[dataset[0]][:-1][int(feature_entry[0])]
                        category_sensitive_discrim_factor = feature_sensitive_percentage[cur_column][labeled_data.iloc[data_point][cur_column]] - percentages_averages[cur_column]
                        model_discrim_factor = feature_entry[1]
                        feature_discrimination_impact[cur_column]+= category_sensitive_discrim_factor*model_discrim_factor
                
                # Logic: absolute value makes all value positive. We take max for the feature that is causing most unfairness. Return negative, because dual annealing minimizes the function
                # and we want to find points in the feature space that are clearly indicitive of unfairness.
                if reason:
                    max_num = list(feature_discrimination_impact.values())[0]
                    reason = list(feature_discrimination_impact.keys())[0]
                    for feature in feature_discrimination_impact.keys():
                        if feature_discrimination_impact[feature] > max_num:
                            max_num = abs(feature_discrimination_impact[feature])
                            reason = feature
                    return -max_num/NUM_MCMC_SAMPLES, reason
                return -max([abs(e) for e in list(feature_discrimination_impact.values())])/NUM_MCMC_SAMPLES

            max_function_call = st.number_input("How many trials are we running",min_value=1, max_value=100000, value=1000)
            best_points = dual_annealing(sensitive_discrimination_score, [(0, X.shape[0])]*NUM_MCMC_SAMPLES,no_local_search=True, maxfun=max_function_call)
            print(best_points)
            random_points = labeled_X_np[list(map(int, best_points.x))]
            # random_counter_points = labeled_X_np[random.sample(prediction_probabilities[prediction_probabilities[f'counter_factual_{studying_feature}']].index.tolist(), k=num_points)]
            reasoning = sensitive_discrimination_score(best_points.x, reason=True)
            st.write(f"Features that cause most unfairness: {reasoning[1]}")
            st.write(f"Unfairness score: {reasoning[0]}")

            exp_reg_weight_list = []
            count = 0
            for row in random_points:
                st.write(f"Picked point {list(map(int, best_points.x))[count]}")
                exp_reg = explainer.explain_instance(row, predict_fn, num_features=len(columns[dataset[0]][:-1]), num_samples=50000)
                explaination_fig = as_pyplot_figure(exp_reg.as_list(), list(class_names))
                st.pyplot(explaination_fig)
                count += 1

            exp_counter_weight_list = []
            # for row in random_counter_points:
            #     exp_counter = explainer.explain_instance(row, predict_fn, num_features=len(columns[dataset[0]][:-1]), num_samples=50000)
            #     exp_counter_weight_list.append({columns[dataset[0]][:-1][int(feature[0])]:abs(feature[1]) for feature in exp_counter.as_map()[1]})

            exp_reg_weight_frame = pd.DataFrame.from_records(exp_reg_weight_list)
            exp_counter_weight_frame = pd.DataFrame.from_records(exp_counter_weight_list)
            st.write(exp_counter_weight_frame.mean(axis=0))
            # averaged_reg_figure = plt.bar(exp_reg_weight_frame.mean(axis=0), title=f"LIME {num_points} randomly sampled average absolute probability change for each feature")
            # averaged_counter_figure = plt.bar(exp_counter_weight_frame.mean(axis=0), title=f"LIME {num_points} coutnerfactual randomly sampled average absolute probability change for each feature")
            # plotly_events(averaged_reg_figure)
            # plotly_events(averaged_counter_figure)
        else:
            explain_sample = selected_explain_points_counterfactuals[0]['x']
            sample_point = X[explain_sample:explain_sample+1]
            
            st.write(labeled_data[explain_sample:explain_sample+1])
            st.write(f"Prediction probability: {selected_explain_points_counterfactuals[0]['y']}")


            st.write(f"Prediction: {int_to_cat_labels_map(dataset[0])['label'][str(trained_model.predict(sample_point)[0])]}")
            print(f"original sample point: {sample_point}")
            st.write("Prediction correct" if trained_model.predict(sample_point) == Y[explain_sample:explain_sample+1] else "Prediction Incorrect")
            explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = columns[dataset[0]][:-1],class_names=list(class_names),
                                                    categorical_features=categorical_features[dataset[0]][:-1], 
                                                    categorical_names=categorical_names, random_state=1, kernel_width=3)
            predict_fn = lambda x: trained_model.predict_proba(x)
            print(f"original label encoded {labeled_X_np[explain_sample]}")
            exp = explainer.explain_instance(labeled_X_np[explain_sample], predict_fn, num_features=len(columns[dataset[0]][:-1]), num_samples=50000)
            explaination_fig = as_pyplot_figure(exp.as_list(), list(class_names))
            st.pyplot(explaination_fig)





            # Counter-factual study (manually change the weights)
            
            st.write("Counter-factuals study")

            counter_factuals_labeled_data = []
            rows = []
            n_cols = 5
            for feature_index in range(len(columns[dataset[0]][:-1])):
                if feature_index%n_cols == 0:
                    rows.append(st.columns(n_cols))
                feature = columns[dataset[0]][:-1][feature_index]
                if feature_index in categorical_features[dataset[0]][:-1]:
                    possible_options = list(int_to_cat_labels_map(dataset[0])[feature].values())
                    counter_factuals_labeled_data.append(
                        rows[math.floor(feature_index/n_cols)][feature_index%n_cols].selectbox(feature, possible_options, index=possible_options.index(labeled_data[explain_sample:explain_sample+1].values.tolist()[0][feature_index]))
                    )
                else:
                    counter_factuals_labeled_data.append(
                        rows[math.floor(feature_index/n_cols)][feature_index%n_cols].number_input(feature,value=labeled_data[explain_sample:explain_sample+1].values.tolist()[0][feature_index])
                    )

            counter_factuals_labeled_data = np.array([counter_factuals_labeled_data], dtype=object)
            counter_factuals_data = counter_factuals_labeled_data.copy()
            for feature in categorical_features[dataset[0]][:-1]:
                counter_factuals_data[:, feature] = feature_label_encoders[feature].transform(counter_factuals_labeled_data[:, feature])
            print(f"counter-factual label encoded: {counter_factuals_data}")
            predict_fn = lambda x: trained_model.predict_proba(x)
            explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = columns[dataset[0]][:-1],class_names=list(class_names),
                                                categorical_features=categorical_features[dataset[0]][:-1], 
                                                categorical_names=categorical_names, random_state=1, kernel_width=3) # This de-randomizes the state, so that the two explained will be identical if the inputs are identical
                                                                                                                    # This helps remove doubt/confusion.
            exp = explainer.explain_instance(counter_factuals_data[0], predict_fn, num_features=len(columns[dataset[0]][:-1]), num_samples=50000)
            explaination_fig = as_pyplot_figure(exp.as_list(), list(class_names))
            
            sample_point = counter_factuals_labeled_data.copy()
            for feature in categorical_features[dataset[0]][:-1]:
                sample_point[0, feature] = int(cat_to_int_map(dataset[0])[columns[dataset[0]][feature]][sample_point[0,feature]])
            print(f"counter-factual sample points: {sample_point}")
            st.write(f"Prediction: {int_to_cat_labels_map(dataset[0])['label'][str(trained_model.predict(sample_point)[0])]}")
            st.write(f"Prediction probability: {str(max(trained_model.predict_proba(sample_point)[0]))}")
            st.write("Prediction correct" if trained_model.predict(sample_point) == Y[explain_sample:explain_sample+1] else "Prediction Incorrect")
            
            st.pyplot(explaination_fig)



# Code taken from https://github.com/marcotcr/lime/blob/master/lime/explanation.py, with updates to deal with differing color schemes, etc.
def as_pyplot_figure(exp, class_names, label=1, figsize=(4,4), title=""):
    """Returns the explanation as a pyplot figure.
    Will throw an error if you don't have matplotlib installed
    Args:
        label: desired label. If you ask for a label for which an
                explanation wasn't computed, will throw an exception.
                Will be ignored for regression explanations.
        figsize: desired size of pyplot in tuple format, defaults to (4,4).
        kwargs: keyword arguments, passed to domain_mapper
    Returns:
        pyplot figure (barchart).
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    vals = [x[1] for x in exp]
    names = [x[0] for x in exp]
    vals.reverse()
    names.reverse()
    colors = ['blue' if x > 0 else 'red' for x in vals]
    pos = np.arange(len(exp)) + .5
    plt.barh(pos, vals, align='center', color=colors)
    plt.yticks(pos, names)
    title = f'Local explanation for class {class_names[label]} {title}'
    plt.title(title)
    return fig
        




def main():
    models = ["LR","RF","SV","DT"]
    models = ["Logistic Regression","Random Forest","Support Vector Machine","Decision Tree"]

    models_key = {"Logistic Regression":"LR", "Random Forest": "RF", "Support Vector Machine": "SV", "Decision Tree": "DT"}
    # Note: Index starts at 1 for the datasets, so subtract 1 from datasets[2] to ensure we are highlighting the correct sensitive feature!
    datasets = [("census", "gender",9), ("census", "race",8), ("credit", "gender",9), ("bank","age",1), ("compas","gender",1), ("compas","race",3)]

    algorithms = ["mutation"]

    # Basic buttons to choose the correct models
    
    picked_dataset = st.radio("Dataset:", datasets, horizontal = True)
    picked_model = st.radio("Model:", models, horizontal = True)
    picked_algo = st.radio("Algorithm:", algorithms, horizontal = True)
    create_model(models_key[picked_model], picked_dataset, picked_algo)