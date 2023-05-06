import pandas as pd
import pickle
import pathlib
import os

import plotly.express as plt
import plotly.tools as tools
import plotly.graph_objects as go
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from matplotlib import pyplot as mplt
import sklearn
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import math
import numpy as np
import lime
import lime.lime_tabular
from scipy.optimize import dual_annealing
from configs import columns, get_groups, labeled_df, categorical_features, categorical_features_names, int_to_cat_labels_map, cat_to_int_map
from home_metrics_helper import model_accuracy, AOD_score, EOD_score
from subjects.adf_data.census import census_data
from subjects.adf_data.credit import credit_data
from subjects.adf_data.bank import bank_data
from subjects.adf_data.compas import compas_data
data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}
def main():
    datasets = [("credit", "gender",9),("census", "race",8), ("census", "gender",9), ("bank","age",1), ("compas","gender",1), ("compas","race",3)]
    models = ["LR","RF","SV","DT"]
    for data in datasets:
        for mod in models:
            
            
            dataset = open(f"data/{data[0]}.csv", "rb")
            org_input_dataset = pd.read_csv(dataset) # Keep a copy untouched!
            encoder = open(f"data/encoded_{data[0]}.csv", "r")
            org_encoded_dataset = pd.read_csv(encoder).to_numpy()
            model = open(f"data/{mod}_{data[0]}_{data[1]}_most_accurate.pkl", "rb")
            input_model = pickle.load(model)
            os.mkdir(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/")
            os.mkdir(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/hist/")
            model_scores = []
            masking_data = []

            input_dataset = org_input_dataset.copy()
            encoded_dataset = org_encoded_dataset.copy()

            for iterations in range(5):
                # The data analysis part
                if dataset is not None:
                    

                    columns = list(input_dataset.columns)

                    categorical_indexes = categorical_features[data[0]]
                    categorical_data = [columns[i] for i in categorical_features[data[0]]]


                    label = "label"
                    label_index = list(columns).index(label)

                    # This really puts another limit on what types of data we can support!
                    sensitive_attribute = "sex" if data[1] == "gender" else data[1]
                    sensitive_attribute_index = list(columns).index(sensitive_attribute)
                    group0, group1 = get_groups(data[0], sensitive_attribute, get_name=True)
                    group0 = [group0]
                    group1 = [group1]
                    
                    


                    X = np.delete(encoded_dataset, label_index, 1)
                    y = encoded_dataset[:,label_index]
                    labeled_X_np = input_dataset.drop(columns=[label]).to_numpy()
                    labeled_Y_np = input_dataset[label].to_numpy()
                    trainX, testX, trainY, testY, labeledTrainX, labeledTestX, labeledTrainY, labeledTestY, inputDatasetTrain, inputDatasetTest, encodedTrain, encodedTest, orgInputDatasetTrain, orgInputDatasetTest = \
                        sklearn.model_selection.train_test_split(X, y, labeled_X_np, labeled_Y_np, input_dataset, encoded_dataset, org_input_dataset, train_size=0.80, random_state=1)
                    

                    X_transformed = StandardScaler().fit_transform(encodedTrain)
                    cov = EmpiricalCovariance().fit(X_transformed)
                    bar_data = {"Column number": columns, "Correlation": cov.covariance_[sensitive_attribute_index], "sensitive_feature": [False]*len(columns)}
                    bar_data["sensitive_feature"][sensitive_attribute_index] = True
                    bar_data = pd.DataFrame(bar_data)
                    fig = plt.bar(bar_data, x="Column number", y="Correlation", color="sensitive_feature", category_orders={"Column number": columns},color_discrete_map={False:'blue', True:'red'}, title=f"Correlation of each column with {sensitive_attribute}")
                    fig.write_image(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/correlation_{iterations}.png")




                


                # The model analysis part
                
                if dataset is not None and model is not None and len(group0)>0 and len(group1)>0:
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
                        fig_2.write_image(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/model_logic_{iterations}.png")
                        fig_3.write_image(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/model_logic_stdadj_{iterations}.png")
                    elif type(input_model).__name__ == "DecisionTreeClassifier":
                        from sklearn import tree
                        import graphviz as graphviz
                        max_depth = 2
                        fig_tree = mplt.figure(figsize=(25,20))
                        _ = tree.plot_tree(input_model, feature_names = columns_without_label,class_names=list(class_names), filled=True)
                        fig_tree.savefig(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/model_logic_{iterations}.png", dpi='figure', format=None)
                    elif type(input_model).__name__ == "LogisticRegression":
                        model_df = {"feature": columns_without_label, "weight":input_model.coef_[0], "std_adj_weights":input_model.coef_[0]*np.std(np.delete(encodedTrain, label_index, 1), axis=0), "sensitive_feature": [False]*len(input_model.coef_[0])}
                        model_df["sensitive_feature"][sensitive_attribute_index] = True

                        fig_2 = plt.bar(model_df, x="feature", y="weight", color="sensitive_feature", category_orders={"feature": columns_without_label},color_discrete_map={False:'blue', True:'red'},
                            title="The LR Model's coefficient/weights")
                        fig_3 = plt.bar(model_df, x="feature", y="std_adj_weights", color="sensitive_feature", category_orders={"feature": columns_without_label},color_discrete_map={False:'blue', True:'red'},
                            title="The LR Model's standard deviation adjusted weights")
                        fig_2.write_image(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/model_logic_{iterations}.png")
                        fig_3.write_image(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/model_logic_stdadj_{iterations}.png")
                    elif type(input_model).__name__ == "RandomForestClassifier":
                        from sklearn import tree
                        import graphviz as graphviz
                    else:
                        raise Exception("Wrong model")
                    # Some sample elements

                    NUM_MCMC_SAMPLES = 3

                    feature_sensitive_percentage = {}
                    percentages_averages = {}


                    
                    explainer = lime.lime_tabular.LimeTabularExplainer(trainX ,feature_names = columns_without_label,class_names=list(class_names),
                                                            categorical_features=categorical_X_index, 
                                                            categorical_names=categorical_names, random_state=1, kernel_width=3)
                    predict_fn = lambda x: input_model.predict_proba(x)

                    feature_sensitive_percentage0 = {}
                    feature_sensitive_percentage1 = {}
                    for col in columns:

                        group_0_val_count = inputDatasetTrain[col][(orgInputDatasetTrain[sensitive_attribute].isin(group0))].value_counts().fillna(0)
                        group_1_val_count = inputDatasetTrain[col][(orgInputDatasetTrain[sensitive_attribute].isin(group1))].value_counts().fillna(0)


                        feature_sensitive_percentage0[col] = (group_0_val_count/(group_0_val_count.add(group_1_val_count, fill_value=0))*100).fillna(0)
                        feature_sensitive_percentage1[col] = (group_1_val_count/(group_0_val_count.add(group_1_val_count, fill_value=0))*100).fillna(0)
                        
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
                                if inputDatasetTrain.iloc[data_point][cur_column] in feature_sensitive_percentage0[cur_column]:
                                    category_sensitive_discrim_factor = feature_sensitive_percentage0[cur_column][inputDatasetTrain.iloc[data_point][cur_column]] - percentages_averages[cur_column]
                                else:
                                    category_sensitive_discrim_factor = 0 # This point is what we are looking for.
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


                    max_function_call = 50
                    suggest_feature_to_hide = []

                    best_points = dual_annealing(sensitive_discrimination_score, [(0, trainX.shape[0])]*NUM_MCMC_SAMPLES,no_local_search=True, maxfun=max_function_call)
                    print(best_points)
                    random_points = labeledTrainX[list(map(int, best_points.x))]
                    # random_counter_points = labeledTrainX[random.sample(prediction_probabilities[prediction_probabilities[f'counter_factual_{studying_feature}']].index.tolist(), k=num_points)]
                    reasoning = sensitive_discrimination_score(best_points.x, reason=True)
                    suggest_feature_to_hide = reasoning[1]
                    write_reasoning = open(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/reasoning.txt", "a")

                    write_reasoning.write(str(reasoning)+"\n")
                    write_reasoning.close()

                    exp_reg_weight_list = []
                    count = 0
                    for row in random_points:
                        exp_reg = explainer.explain_instance(row, predict_fn, num_features=len(columns_without_label), num_samples=50000)
                        explaination_fig = as_pyplot_figure(exp_reg.as_list(), list(class_names))
                        explaination_fig.savefig(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/explaination_fig_{iterations}_{count}.png", dpi='figure', format=None)
                        count += 1

                    exp_counter_weight_list = []

                    # This allows for us to set the multiselect below to our desired "default" values

                    

                    # Calculate avaliable metrics
                    model_accuracy_score = model_accuracy(input_model, X, y)
                    
                    encodedTest_group0 = encodedTest[orgInputDatasetTest[sensitive_attribute].isin(group0)] # Should match up in theory...

                    X_group0 = np.delete(encodedTest_group0, label_index, 1)
                    y_group0 = encodedTest_group0[:,label_index]
                    encodedTest_group1 = encodedTest[orgInputDatasetTest[sensitive_attribute].isin(group1)] # Should match up in theory...
                    
                    X_group1 = np.delete(encodedTest_group1, label_index, 1)
                    y_group1 = encodedTest_group1[:,label_index]
                    model_AOD_score = AOD_score(input_model, X_group0, y_group0, X_group1, y_group1)
                    model_EOD_score = EOD_score(input_model, X_group0, y_group0, X_group1, y_group1)


                    # No longer saving the models
                    # st.session_state["saved_models"].append(pickle.dumps(input_model))

                    write_scores = open(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/scores.txt", "a")

                    write_scores.write(str([model_accuracy_score, model_AOD_score, model_EOD_score])+"\n")
                    write_scores.close()

                    model_scores.append([model_accuracy_score, model_AOD_score, model_EOD_score])
                    
                    model_scores_pd = pd.DataFrame(model_scores, columns=["Accuracy", "AOD", "EOD"])
                    fig_scores = go.Figure()
                    fig_scores.add_trace(go.Scatter(x=model_scores_pd.index, y=model_scores_pd["Accuracy"],
                                        mode='lines',
                                        name='Accuracy'))
                    fig_scores.add_trace(go.Scatter(x=model_scores_pd.index, y=model_scores_pd["AOD"],
                                        mode='lines',
                                        name='Average Odds Difference'))
                    fig_scores.add_trace(go.Scatter(x=model_scores_pd.index, y=model_scores_pd["EOD"],
                                        mode='lines',
                                        name='Equalized Odds Difference'))
                    fig_scores.update_layout(title='Fairness and Accuracy Scores of Previous Models',
                                    xaxis_title='Iteration',
                                    yaxis_title='Model Score')
                    fig_scores.write_image(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/fig_scores_{iterations}.png")

                    
                    # Provide suggested remedies


                    masking_data.append(suggest_feature_to_hide)
                    categorical_remedy = {}
                    remedy = {}
                    for masking in masking_data:
                        if masking in categorical_data:
                            masking_cat = categorical_remedy[masking] if masking in categorical_remedy else list(pd.unique(org_input_dataset[masking]))
                            categorical_remedy[masking] = masking_cat

                    input_dataset = org_input_dataset.copy()
                    encoded_dataset = org_encoded_dataset.copy()
                    for masking in masking_data:
                        remedy[masking] = None
                        if masking in categorical_data:
                            masking_cat = categorical_remedy[masking]
                            remedy[masking] = masking_cat
                            input_dataset[masking][org_input_dataset[masking].isin(masking_cat)] = "Masked"
                            encoded_dataset[org_input_dataset[masking].isin(masking_cat),list(columns).index(masking)] = 0 # Should match up in theory...

                        else:
                            input_dataset[masking] = 0
                            encoded_dataset[:,list(columns).index(masking)] = 0 # Should match up in theory...



                    input_model = input_model.fit(trainX, trainY)
                        


                    for col in columns:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Histogram(x=inputDatasetTrain[col][(orgInputDatasetTrain[sensitive_attribute].isin(group0))],histnorm="", histfunc="count", name=f'{group0}'))
                        fig2.add_trace(go.Histogram(x=inputDatasetTrain[col][(orgInputDatasetTrain[sensitive_attribute].isin(group1))],histnorm="", histfunc="count", name=f'{group1}'))
                        fig2.update_layout(
                            title_text=f"Histogram of {sensitive_attribute} v.s. {col}", # title of plot
                            xaxis_title_text=f"{col}", # xaxis label
                            yaxis_title_text='Count', # yaxis label
                            bargap=0.2, # gap between bars of adjacent location coordinates
                            bargroupgap=0.1 # gap between bars of the same location coordinates
                        )

                        group_0_count = inputDatasetTrain[col][(orgInputDatasetTrain[sensitive_attribute].isin(group0))].value_counts(dropna=False).fillna(0)
                        group_1_count = inputDatasetTrain[col][(orgInputDatasetTrain[sensitive_attribute].isin(group1))].value_counts(dropna=False).fillna(0)

                        feature_sensitive_percentage0 = group_0_count/(group_0_count.add(group_1_count, fill_value=0))*100
                        feature_sensitive_percentage1 = group_1_count/(group_0_count.add(group_1_count, fill_value=0))*100


                        # feature_sensitive_percentage = feature_sensitive_percentage.rename(group0)
                        fig3 = plt.bar(feature_sensitive_percentage0, title=f"Histogram of % {group0} v.s. {col} scaled")
                        fig3.add_hline(y=inputDatasetTrain[orgInputDatasetTrain[sensitive_attribute].isin(group0)][sensitive_attribute].count()/inputDatasetTrain[sensitive_attribute].count()*100, line_dash="dot",annotation_text="Average percentage")
                        
                        fig3.update_layout(
                            title_text=f"Histogram of % {group0} in {col} scaled", # title of plot
                            xaxis_title_text=f"{col}", # xaxis label
                            yaxis_title_text='Percentage', # yaxis label
                            bargap=0.2, # gap between bars of adjacent location coordinates
                            bargroupgap=0.1 # gap between bars of the same location coordinates
                        )
                        fig4 = plt.bar(feature_sensitive_percentage1, title=f"Histogram of % {group1} v.s. {col} scaled")
                        fig4.add_hline(y=inputDatasetTrain[orgInputDatasetTrain[sensitive_attribute].isin(group1)][sensitive_attribute].count()/inputDatasetTrain[sensitive_attribute].count()*100, line_dash="dot",annotation_text="Average percentage")
                        
                        fig4.update_layout(
                            title_text=f"Histogram of % {group1} in {col} scaled", # title of plot
                            xaxis_title_text=f"{col}", # xaxis label
                            yaxis_title_text='Percentage', # yaxis label
                            bargap=0.2, # gap between bars of adjacent location coordinates
                            bargroupgap=0.1 # gap between bars of the same location coordinates
                        )    

                        if is_numeric_dtype(inputDatasetTrain[col]):
                            fig2.update_xaxes(categoryorder='array', categoryarray=list(inputDatasetTrain[col]))
                            fig3.update_xaxes(categoryorder='array', categoryarray=list(inputDatasetTrain[col]))
                            fig4.update_xaxes(categoryorder='array', categoryarray=list(inputDatasetTrain[col]))
                        else:
                            fig2.update_xaxes(categoryorder='category ascending')
                            fig3.update_xaxes(categoryorder='category ascending')
                            fig4.update_xaxes(categoryorder='category ascending')
                        
                        
                        fig2.write_image(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/hist/histograph_{iterations}.png")
                        fig3.write_image(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/hist/histograph_g0_percent_{col}_{iterations}.png")
                        fig4.write_image(f"generated_metrics_and_graphics/{data[0]}_{data[1]}_{mod}/hist/histograph_g1_percent_{col}_{iterations}.png")



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

main()