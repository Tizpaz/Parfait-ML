import streamlit as st
import pandas as pd
import pickle
import pathlib
import os

import plotly.express as plt
import plotly.tools as tools
import plotly.graph_objects as go
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

import sklearn
from streamlit_plotly_events import plotly_events
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import math
import numpy as np
import lime
import lime.lime_tabular
from scipy.optimize import dual_annealing
from home_metrics_helper import model_accuracy, AOD_score, EOD_score
def main():
    st.write("Hello! My name is Normen Yu, a student at Pennsylvania State University. This web application is inspired by my bachelor thesis advisor Dr. Gan Tan and Dr. Saeid Tizpaz-Niari’s research on strategies to mitigate unfairness in machine learning algorithms. As machine learning becomes more prevalent in use for critical social decisions such as court systems, loan approval systems, and predicting income, so too does its potential impact if they are fundamentally unjust or unfair. Hence, fairness in machine learning has become a topic of hot discussion and research. Personally, I believe that bias in machine learning algorithms is fundamentally a human construct, and therefore requires humans to intervene on a case-by-case basis. To do this require in-depth understanding of 1) the trade-offs between and accuracy/efficiency to the general population, 2) thorough understanding of why a model made a specific decision, and 3) the historic nature of bias in pre-existing data. This application provides the framework and a specific proof-of-concept to visualize these complex logic in an intuitive fashion.")
    st.write("This project leverages some industrial tools I have used during my internships (Docker, Streamlit, Plotly, etc) to create an application that can help researchers analyze and visualize bias in their learned models. These tools utilized are selected because they require minimum set-up: any developer with Python/pip set-up can easily spin-up my scripts on https://github.com/Pennswood/Parfait-ML/tree/live-streamlit-app, make modifications, and adapt to each researcher’s use case. With just a streamlit command and a python script, there is no heavy back-end set-up to spin up a localhost web-app! Ultimately, the goal of this project is to inspire even those with non-technical background – political scientists, decision makers, social workers, businessman, and so many more with low-to-no understanding of machine learning or statistics – to be able to use this tool and understand intuitively what is happening (the 'means') and what is at stake (the 'ends').")



    
    dataset =st.file_uploader("Upload your dataset here", type="csv")

    encoder =st.file_uploader("Upload your SkLearn encoder or encoded dataset here (if none provided, we will encode your data with Ordinal encoder", type=["pkl", "pickle", "csv"])



    # The data analysis part
    if dataset is not None:
        st.title("Dataset Analysis")
        
        if "input_dataset" not in st.session_state:
            st.session_state["org_input_dataset"] = pd.read_csv(dataset) # Keep a copy untouched!
            input_dataset = st.session_state["org_input_dataset"].copy()
        else:
            input_dataset = st.session_state["input_dataset"]
        columns = list(input_dataset.columns)
        # We need to know what are numerical data an what are categorical data
        with st.expander("Specify Your Dataset", expanded=True):
            categorical_data = []
            categorical_indexes = []
            counter = 0
            for col in columns:
                if is_string_dtype(input_dataset[col]):
                    categorical_data.append(col)
                    categorical_indexes.append(counter)
                elif len(pd.unique(input_dataset[col])) < 10:
                    categorical_data.append(col)
                    categorical_indexes.append(counter)
                counter += 1
            categorical_data = st.multiselect("Please verify the categorical data",columns, default=categorical_data)
            numerica_features = list(set(columns)-set(categorical_data))
            st.write(f"Numerical features: {numerica_features}")

            label = st.selectbox("Please verify the label (prediction) column (Only categorical predictions supported at this time)",categorical_data, index=len(categorical_data)-1)
            label_index = list(columns).index(label)

            # This really puts another limit on what types of data we can support!
            sensitive_attribute = st.selectbox("Pick your sensitive attribute (only categorical data allowed)",categorical_data)
            sensitive_attribute_index = list(columns).index(sensitive_attribute)
            sensitive_categories = st.session_state["org_input_dataset"][sensitive_attribute].unique() # Use original if user wants to mask the sensitive categoriy itself, we still need to pick the sensitive categories
            group0 = st.multiselect("Select categories for group 0", sensitive_categories)
            group1 = st.multiselect("Select categories for group 1", list(set(sensitive_categories) - set(group0)))
        
        if "encoded_dataset" not in st.session_state:
            if encoder is not None:
                _, encoder_extension = os.path.splitext(encoder.name)
                if encoder_extension.lower() == ".pkl" or encoder_extension.lower() == ".pickle":
                    input_encoder = pickle.load(encoder)
                    encoded_dataset = input_encoder().fit_transform(input_dataset)
                elif encoder_extension.lower() == ".csv":
                    encoded_dataset = pd.read_csv(encoder).to_numpy()
                    if encoded_dataset.shape[0] != input_dataset.shape[0]:
                        raise Exception("Your encoded dataset row count does not match your original input dataset size!")
                    if encoded_dataset.shape[1] != input_dataset.shape[1]:
                        raise Exception("Your encoded dataset column count does not match your original input dataset size!")
            else:
                encoded_dataset = OrdinalEncoder().fit_transform(input_dataset)
            st.session_state["org_encoded_dataset"] = encoded_dataset.copy()
        else:
            encoded_dataset = st.session_state["encoded_dataset"]


        with st.expander("Visualize the Dataset"):
            X = np.delete(encoded_dataset, label_index, 1)
            y = encoded_dataset[:,label_index]
            labeled_X_np = input_dataset.drop(columns=[label]).to_numpy()
            labeled_Y_np = input_dataset[label].to_numpy()
            trainX, testX, trainY, testY, labeledTrainX, labeledTestX, labeledTrainY, labeledTestY, inputDatasetTrain, inputDatasetTest, encodedTrain, encodedTest, orgInputDatasetTrain, orgInputDatasetTest = \
                sklearn.model_selection.train_test_split(X, y, labeled_X_np, labeled_Y_np, input_dataset, encoded_dataset, st.session_state["org_input_dataset"], train_size=0.80, random_state=1)
            

            X_transformed = StandardScaler().fit_transform(encodedTrain)
            cov = EmpiricalCovariance().fit(X_transformed)
            bar_data = {"Column number": columns, "Correlation": cov.covariance_[sensitive_attribute_index], "sensitive_feature": [False]*len(columns)}
            bar_data["sensitive_feature"][sensitive_attribute_index] = True
            bar_data = pd.DataFrame(bar_data)
            st.write("Note: Many of the data are categorical, making correlation a bad metric. Click on the bar to see more detailed labelled bar/histogram graphs.")
            fig = plt.bar(bar_data, x="Column number", y="Correlation", color="sensitive_feature", category_orders={"Column number": columns},color_discrete_map={False:'blue', True:'red'}, title=f"Correlation of each column with {sensitive_attribute}")
            bar_graph = plotly_events(fig)
            


            
            if bar_graph != [] and len(group0)>0 and len(group1) >0:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=inputDatasetTrain[bar_graph[0]["x"]][(orgInputDatasetTrain[sensitive_attribute].isin(group0))],histnorm="", histfunc="count", name=f'{group0}'))
                fig2.add_trace(go.Histogram(x=inputDatasetTrain[bar_graph[0]["x"]][(orgInputDatasetTrain[sensitive_attribute].isin(group1))],histnorm="", histfunc="count", name=f'{group1}'))
                fig2.update_layout(
                    title_text=f"Histogram of {sensitive_attribute} v.s. {bar_graph[0]['x']}", # title of plot
                    xaxis_title_text=f"{bar_graph[0]['x']}", # xaxis label
                    yaxis_title_text='Count', # yaxis label
                    bargap=0.2, # gap between bars of adjacent location coordinates
                    bargroupgap=0.1 # gap between bars of the same location coordinates
                )

                group_0_count = inputDatasetTrain[bar_graph[0]["x"]][(orgInputDatasetTrain[sensitive_attribute].isin(group0))].value_counts(dropna=False).fillna(0)
                group_1_count = inputDatasetTrain[bar_graph[0]["x"]][(orgInputDatasetTrain[sensitive_attribute].isin(group1))].value_counts(dropna=False).fillna(0)
                feature_sensitive_percentage0 = group_0_count/(group_0_count.add(group_1_count, fill_value=0))*100
                feature_sensitive_percentage1 = group_1_count/(group_0_count.add(group_1_count, fill_value=0))*100


                # feature_sensitive_percentage = feature_sensitive_percentage.rename(group0)
                fig3 = plt.bar(feature_sensitive_percentage0, title=f"Histogram of % {group0} v.s. {bar_graph[0]['x']} scaled")
                fig3.add_hline(y=inputDatasetTrain[orgInputDatasetTrain[sensitive_attribute].isin(group0)][sensitive_attribute].count()/inputDatasetTrain[sensitive_attribute].count()*100, line_dash="dot",annotation_text="Average percentage")
                
                fig3.update_layout(
                    title_text=f"Histogram of % {group0} in {bar_graph[0]['x']} scaled", # title of plot
                    xaxis_title_text=f"{bar_graph[0]['x']}", # xaxis label
                    yaxis_title_text='Percentage', # yaxis label
                    bargap=0.2, # gap between bars of adjacent location coordinates
                    bargroupgap=0.1 # gap between bars of the same location coordinates
                )
                fig4 = plt.bar(feature_sensitive_percentage1, title=f"Histogram of % {group1} v.s. {bar_graph[0]['x']} scaled")
                fig4.add_hline(y=inputDatasetTrain[orgInputDatasetTrain[sensitive_attribute].isin(group1)][sensitive_attribute].count()/inputDatasetTrain[sensitive_attribute].count()*100, line_dash="dot",annotation_text="Average percentage")
                
                fig4.update_layout(
                    title_text=f"Histogram of % {group1} in {bar_graph[0]['x']} scaled", # title of plot
                    xaxis_title_text=f"{bar_graph[0]['x']}", # xaxis label
                    yaxis_title_text='Percentage', # yaxis label
                    bargap=0.2, # gap between bars of adjacent location coordinates
                    bargroupgap=0.1 # gap between bars of the same location coordinates
                )    

                if is_numeric_dtype(inputDatasetTrain[bar_graph[0]["x"]]):
                    fig2.update_xaxes(categoryorder='array', categoryarray=list(inputDatasetTrain[bar_graph[0]["x"]]))
                    fig3.update_xaxes(categoryorder='array', categoryarray=list(inputDatasetTrain[bar_graph[0]["x"]]))
                    fig4.update_xaxes(categoryorder='array', categoryarray=list(inputDatasetTrain[bar_graph[0]["x"]]))
                else:
                    fig2.update_xaxes(categoryorder='category ascending')
                    fig3.update_xaxes(categoryorder='category ascending')
                    fig4.update_xaxes(categoryorder='category ascending')
                
                hist_graph = plotly_events(fig2)
                percent_graph1 = plotly_events(fig3)
                percent_graph2 = plotly_events(fig4)

    model = st.file_uploader("Upload your SkLearn model here", type=["pkl", "pickle"])


    # The model analysis part
    
    if model is not None:
        st.title("Model and Explanability Analysis")
        if "updated_model" in st.session_state and st.session_state["updated_model"]:
            input_model = st.session_state["updated_model"]
        else:
            input_model = pickle.load(model)
        st.write(type(input_model).__name__)
        hyperparams = input_model.get_params()
        n_cols = 4
        checked = []
        rows = []
        i = 0
        with st.expander("See Hyperparameters"):
            for hyperparam_name, hyperparam_val in hyperparams.items():
                if i%n_cols == 0:
                    rows.append(st.columns(n_cols))
                checked.append(rows[math.floor(i/n_cols)][i%n_cols].text_input(hyperparam_name, value=hyperparam_val, disabled=True))
                i+= 1
        


    if dataset is not None and model is not None and len(group0)>0 and len(group1)>0:
        
        columns_without_label = columns.copy()
        columns_without_label.pop(label_index)
        # Display the models
        if type(input_model).__name__ == "SVC":
            model_df = {"feature": columns_without_label, "weight":input_model.coef_[0], "std_adj_weights":input_model.coef_[0]*np.std(np.delete(encodedTrain, label_index, 1), axis=0), "sensitive_feature": [False]*len(input_model.coef_[0])}
            model_df["sensitive_feature"][sensitive_attribute_index] = True
            fig_2 = plt.bar(model_df, x="feature", y="weight", color="sensitive_feature", category_orders={"feature": columns_without_label}, color_discrete_map={False:'blue', True:'red'},
                title="The SVM Model's hyperplane slopes")
            fig_3 = plt.bar(model_df, x="feature", y="std_adj_weights", color="sensitive_feature", category_orders={"feature": columns_without_label},color_discrete_map={False:'blue', True:'red'},
                title="The SVM Model's standard deviation adjusted hyperplane slopes")
            with st.expander("See Model Logic"):
                plotly_events(fig_2)
                plotly_events(fig_3)
        elif type(input_model).__name__ == "DecisionTreeClassifier":
            from sklearn import tree
            import graphviz as graphviz
            max_depth = st.slider('Max depth', 0, 10, 2)
            dot_data = tree.export_graphviz(input_model,max_depth=max_depth, feature_names=columns_without_label, proportion = True, class_names=True, out_file=None, rounded=True)
            with st.expander("See Model Logic"):
                st.graphviz_chart(dot_data)
        elif type(input_model).__name__ == "LogisticRegression":
            model_df = {"feature": columns_without_label, "weight":input_model.coef_[0], "std_adj_weights":input_model.coef_[0]*np.std(np.delete(encodedTrain, label_index, 1), axis=0), "sensitive_feature": [False]*len(input_model.coef_[0])}
            model_df["sensitive_feature"][sensitive_attribute_index] = True

            fig_2 = plt.bar(model_df, x="feature", y="weight", color="sensitive_feature", category_orders={"feature": columns_without_label},color_discrete_map={False:'blue', True:'red'},
                title="The LR Model's coefficient/weights")
            fig_3 = plt.bar(model_df, x="feature", y="std_adj_weights", color="sensitive_feature", category_orders={"feature": columns_without_label},color_discrete_map={False:'blue', True:'red'},
                title="The LR Model's standard deviation adjusted weights")
            with st.expander("See Model Logic"):
                plotly_events(fig_2)
                plotly_events(fig_3)
        elif type(input_model).__name__ == "RandomForestClassifier":
            from sklearn import tree
            import graphviz as graphviz
            
            checked = []
            rows = []
            n_cols = 10
            with st.expander("See Model Logic"):
                st.write("Display tree:")
                for i in range(len(input_model.estimators_)):
                    if i%n_cols == 0:
                        rows.append(st.columns(n_cols))
                    checked.append(rows[math.floor(i/n_cols)][i%n_cols].checkbox(label=f"{i}"))
                
                max_depth = st.slider('Max depth', 0, 10, 2)
                for i in range(len(checked)):
                    if checked[i]:
                        print(columns_without_label)
                        print(input_model.estimators_[i].n_features_in_)
                        dot_data = tree.export_graphviz(input_model.estimators_[i],max_depth=max_depth, feature_names=columns_without_label, proportion = True, class_names=True, out_file=None, rounded=True)
                        st.graphviz_chart(dot_data)
        else:
            st.write("Direct visualization of your model is not yet supported (current support to sci-kit learn's Logistic Regression, Support Vector Classifier, Random Forest, and Decision Tree")

        # Some sample elements

        NUM_MCMC_SAMPLES = 3

        feature_sensitive_percentage = {}
        percentages_averages = {}


        
        categorical_X_data = categorical_data.copy()
        categorical_X_index = categorical_indexes.copy()
        categorical_X_index.pop(categorical_X_data.index(label))
        categorical_X_data.remove(label)


        categorical_names = {}
        feature_label_encoders = {}
        # st.write(st.session_state["input_dataset"])

        for feature in categorical_X_index:
            le_x = sklearn.preprocessing.LabelEncoder()
            le_x.fit(labeledTrainX[:, feature])
            feature_label_encoders[feature] = le_x
            labeledTrainX[:, feature] = le_x.transform(labeledTrainX[:, feature])
            categorical_names[feature] = list(le_x.classes_)
        
        
        
        le_y = sklearn.preprocessing.LabelEncoder()
        le_y.fit(labeledTrainY)
        labeledTrainY = le_y.transform(labeledTrainY)
        class_names = le_y.classes_

        
        explainer = lime.lime_tabular.LimeTabularExplainer(trainX ,feature_names = columns_without_label,class_names=list(class_names),
                                                categorical_features=categorical_X_index, 
                                                categorical_names=categorical_names, random_state=1, kernel_width=3)
        predict_fn = lambda x: input_model.predict_proba(x)


        for col in columns:

            group_0_val_count = inputDatasetTrain[col][(orgInputDatasetTrain[sensitive_attribute].isin(group0))].value_counts().fillna(0)
            group_1_val_count = inputDatasetTrain[col][(orgInputDatasetTrain[sensitive_attribute].isin(group1))].value_counts().fillna(0)


            feature_sensitive_percentage0[col] = group_0_val_count/(group_0_val_count+group_1_val_count)*100
            feature_sensitive_percentage1[col] = group_1_val_count/(group_0_val_count+group_1_val_count)*100
            
            group_0_count = inputDatasetTrain[col][(orgInputDatasetTrain[sensitive_attribute].isin(group0))].count()
            group_1_count = inputDatasetTrain[col][(orgInputDatasetTrain[sensitive_attribute].isin(group1))].count()
            percentages_averages[col] = group_0_count/(group_0_count+group_1_count)*100
        
        def sensitive_discrimination_score(samples, reason = False):
            assert(len(samples) == NUM_MCMC_SAMPLES)
            feature_discrimination_impact = {col:0 for col in columns_without_label}
            for data_point in samples:
                data_point = int(data_point)
                if reason:
                    num_samples=50000
                else:
                    num_samples = 5000

                exp_reg = explainer.explain_instance(labeledTrainX[data_point], predict_fn, num_features=len(columns_without_label), num_samples=num_samples)
                for feature_entry in exp_reg.as_map()[1]:
                    cur_column = columns_without_label[int(feature_entry[0])]
                    category_sensitive_discrim_factor = feature_sensitive_percentage0[cur_column][inputDatasetTrain.iloc[data_point][cur_column]] - percentages_averages[cur_column]
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
        with st.expander("Sampling Some Unfair Examples for LIME"):
            max_function_call = st.number_input("How many trials are we running",min_value=1, max_value=100000, value=500)
            if "suggest_feature_to_hide" not in st.session_state:
                st.session_state["suggest_feature_to_hide"] = []
            if(st.button("Start trials and suggest remedies")):
                st.title("LIME Explanation for Sampled Points")
                st.write("We use LIME and simulated annealing to pick some points from the training dataset that are particularly unfairly classified.")
                best_points = dual_annealing(sensitive_discrimination_score, [(0, trainX.shape[0])]*NUM_MCMC_SAMPLES,no_local_search=True, maxfun=max_function_call)
                print(best_points)
                random_points = labeledTrainX[list(map(int, best_points.x))]
                # random_counter_points = labeledTrainX[random.sample(prediction_probabilities[prediction_probabilities[f'counter_factual_{studying_feature}']].index.tolist(), k=num_points)]
                reasoning = sensitive_discrimination_score(best_points.x, reason=True)
                st.write(f"Features that cause most unfairness: {reasoning[1]}")
                st.session_state["suggest_feature_to_hide"] = [reasoning[1]]
                st.write(f"Unfairness score for the data points: {reasoning[0]}")

                exp_reg_weight_list = []
                count = 0
                for row in random_points:
                    st.write(f"Picked point {list(map(int, best_points.x))[count]}")
                    exp_reg = explainer.explain_instance(row, predict_fn, num_features=len(columns_without_label), num_samples=50000)
                    explaination_fig = as_pyplot_figure(exp_reg.as_list(), list(class_names))
                    st.pyplot(explaination_fig)
                    count += 1

                exp_counter_weight_list = []

                # This allows for us to set the multiselect below to our desired "default" values
                if "counter" not in st.session_state:
                    st.session_state["counter"] = 0
                st.session_state["counter"] += 1

        # Calculate avaliable metrics
        with st.expander("Model Scoring"): # DOne
            model_accuracy_score = model_accuracy(input_model, X, y)
            
            encodedTest_group0 = encodedTest[orgInputDatasetTest[sensitive_attribute].isin(group0)] # Should match up in theory...

            X_group0 = np.delete(encodedTest_group0, label_index, 1)
            y_group0 = encodedTest_group0[:,label_index]
            encodedTest_group1 = encodedTest[orgInputDatasetTest[sensitive_attribute].isin(group1)] # Should match up in theory...
            
            X_group1 = np.delete(encodedTest_group1, label_index, 1)
            y_group1 = encodedTest_group1[:,label_index]
            model_AOD_score = AOD_score(input_model, X_group0, y_group0, X_group1, y_group1)
            model_EOD_score = EOD_score(input_model, X_group0, y_group0, X_group1, y_group1)


            if "model_scores" not in st.session_state:
                st.session_state["model_scores"] = []
                st.session_state["saved_models"] = []
            if "new_model" not in st.session_state or st.session_state["new_model"]:
                st.session_state["model_scores"].append([model_accuracy_score, model_AOD_score, model_EOD_score])
                st.session_state["saved_models"].append(pickle.dumps(input_model))
                st.session_state["new_model"] = False
                
            
            model_scores = pd.DataFrame(st.session_state["model_scores"], columns=["Accuracy", "AOD", "EOD"])
            st.write(model_scores)
            fig_scores = go.Figure()
            fig_scores.add_trace(go.Scatter(x=model_scores.index, y=model_scores["Accuracy"],
                                mode='lines',
                                name='Accuracy'))
            fig_scores.add_trace(go.Scatter(x=model_scores.index, y=model_scores["AOD"],
                                mode='lines',
                                name='Average Odds Difference'))
            fig_scores.add_trace(go.Scatter(x=model_scores.index, y=model_scores["EOD"],
                                mode='lines',
                                name='Equalized Odds Difference'))
            fig_scores.update_layout(title='Fairness and Accuracy Scores of Previous Models',
                            xaxis_title='Iteration',
                            yaxis_title='Model Score')
            picked_model_for_download = plotly_events(fig_scores)
            if len(picked_model_for_download) >0:
                st.download_button(f"Download model {picked_model_for_download[0]['x']}", st.session_state["saved_models"][picked_model_for_download[0]['x']], f"new_model_iteration{picked_model_for_download[0]['x']}.pkl")
            else:
                st.download_button(f"Download model ", "", disabled=True)
        
        # Provide suggested remedies
        st.title("Remedies")
        multiselect_container = st.empty()
        if "remedy" not in st.session_state:
            st.session_state["remedy"] = {}

        # This clears the multiselect, as it uses a different key every time
        if "counter" not in st.session_state:
            st.session_state["counter"] = 0


        masking_data = st.multiselect("Mask the following features",columns,key=st.session_state["counter"], default=list(set(st.session_state["remedy"].keys()).union(set(st.session_state["suggest_feature_to_hide"]))))
        categorical_remedy = {}
        for masking in masking_data:
            if masking in categorical_data:
                masking_cat = st.multiselect(f"Mask the following categories for feature {masking}", list(pd.unique(st.session_state["org_input_dataset"][masking])), \
                    default=categorical_remedy[masking] if masking in categorical_remedy else list(pd.unique(st.session_state["org_input_dataset"][masking])))
                categorical_remedy[masking] = masking_cat
        if st.button("Mask data"):
            st.session_state["suggest_feature_to_hide"] = [] # Clear the "cache"
            st.session_state["input_dataset"] = st.session_state["org_input_dataset"].copy()
            st.session_state["encoded_dataset"] = st.session_state["org_encoded_dataset"].copy()
            for masking in masking_data:
                st.session_state["remedy"][masking] = None
                if masking in categorical_data:
                    masking_cat = categorical_remedy[masking]
                    st.session_state["remedy"][masking] = masking_cat
                    st.session_state["input_dataset"][masking][st.session_state["org_input_dataset"][masking].isin(masking_cat)] = "Masked"
                    st.session_state["encoded_dataset"][st.session_state["org_input_dataset"][masking].isin(masking_cat),list(columns).index(masking)] = 0 # Should match up in theory...

                else:
                    st.session_state["input_dataset"][masking] = 0
                    st.session_state["encoded_dataset"][:,list(columns).index(masking)] = 0 # Should match up in theory...
            
            st.session_state["new_model"] = False

            st.experimental_rerun()
        if st.button("Retrain with masked data"):
            input_model = input_model.fit(trainX, trainY)
            st.session_state["new_model"] = True
            st.session_state["updated_model"] = input_model
            st.experimental_rerun()

        # Save models for user to download



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