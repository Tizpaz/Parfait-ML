from streamlit_plotly_events import plotly_events
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as plt
import plotly.tools as tools
import math
import os
import matplotlib.pyplot as mplt
from configs import columns, get_groups, labeled_df, categorical_features, categorical_features_names
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
            st.write(len(trained_model.coef_))
            model_df = {"feature": columns[dataset[0]][:-1], "weight":trained_model.coef_[0], "sensitive_feature": [False]*len(trained_model.coef_[0])}
            model_df["sensitive_feature"][dataset[2]-1] = True
            fig_2 = plt.bar(model_df, x="feature", y="weight", color="sensitive_feature", category_orders={"feature": columns[dataset[0]][:-1]},color_discrete_map={False:'blue', True:'red'},
                title="The Logistic Regression Model's coefficient/weights for each feature")
            plotly_events(fig_2)
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
            model_df = {"feature": range(len(trained_model.coef_[0])), "weight":trained_model.coef_[0], "sensitive_feature": [False]*len(trained_model.coef_[0])}
            model_df["sensitive_feature"][dataset[2]-1] = True
            fig_2 = plt.bar(model_df, x="feature", y="weight", color="sensitive_feature", category_orders={'sensitive_feature': [False, True]}, color_discrete_sequence=['blue', 'red'],
                title="The Support Vector Machine Model's coefficient/weights for each feature")
            plotly_events(fig_2)

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
        st.write("Explaining a datapoint")
        from subjects.adf_data.census import census_data
        from subjects.adf_data.credit import credit_data
        from subjects.adf_data.bank import bank_data
        from subjects.adf_data.compas import compas_data
        data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}
        X, Y, input_shape, nb_classes = data[dataset[0]]()
        Y = np.argmax(Y, axis=1)
        X = X.astype(np.int64)
        Y = Y.astype(np.int64)
        labeled_data = labeled_df(pd.concat([pd.DataFrame(X),pd.DataFrame(Y)], axis=1), dataset[0])
        labeled_data.columns = columns[dataset[0]]
        explain_sample = st.slider("Pick a data point in the dataset for classification and explaination: ", min_value=0, max_value=int(len(labeled_data.index)-1), step=1)
        sample_point = X[explain_sample:explain_sample+1]
        
        st.write(labeled_data[explain_sample:explain_sample+1])

        import sklearn
        import lime
        import lime.lime_tabular
        from io import BytesIO
        from PIL import Image
        train, test, labeled_train, labeled_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.80)
        categorical_names = {}
        # Processing features so that it fits for LIME
        labeled_X_np = labeled_data.to_numpy()[:,:-1] # to array, minus the label
        for feature in categorical_features[dataset[0]][:-1]:
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(labeled_X_np[:, feature])
            labeled_X_np[:, feature] = le.transform(labeled_X_np[:, feature])
            categorical_names[feature] = list(le.classes_)
        
        
        labeled_Y_np = labeled_data.to_numpy()[:,-1:] # to array, minus the label
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(labeled_Y_np)
        labeled_Y_np = le.transform(labeled_Y_np)
        class_names = le.classes_

        explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = columns[dataset[0]][:-1],class_names=list(class_names),
                                                   categorical_features=categorical_features[dataset[0]][:-1], 
                                                   categorical_names=categorical_names, kernel_width=3)
        predict_fn = lambda x: trained_model.predict_proba(x)
        exp = explainer.explain_instance(labeled_X_np[explain_sample], predict_fn)
        explaination_fig = exp.as_pyplot_figure()
        st.write("Prediction correct" if trained_model.predict(sample_point) == Y[explain_sample:explain_sample+1] else "Prediction Incorrect")
        st.pyplot(explaination_fig)
        

def main():
    models = ["LR","RF","SV","DT"]
    models = ["Logistic Regression","Random Forest","Support Vector Machine","Decision Tree"]

    models_key = {"Logistic Regression":"LR", "Random Forest": "RF", "Support Vector Machine": "SV", "Decision Tree": "DT"}
    # Note: Index starts at 1 for the datasets, so subtract 1 from datasets[2] to ensure we are highlighting the correct sensitive feature!
    datasets = [("census", "gender",9), ("census", "race",8), ("credit", "gender",9), ("bank","age",1), ("compas","gender",1), ("compas","race",3)]

    algorithms = ["mutation"]

    # Basic buttons to choose the correct models
    picked_model = st.radio("Model:", models, horizontal = True)
    picked_dataset = st.radio("Dataset:", datasets, horizontal = True)
    picked_algo = st.radio("Algorithm:", algorithms, horizontal = True)
    create_model(models_key[picked_model], picked_dataset, picked_algo)