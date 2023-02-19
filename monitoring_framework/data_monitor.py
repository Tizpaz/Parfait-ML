from streamlit_plotly_events import plotly_events
import streamlit as st
import numpy as np
import plotly.express as plt
import plotly.tools as tools
import plotly.graph_objects as go
import math
import matplotlib.pyplot as mplt


import sys

sys.path.append("./")
sys.path.append("../")




def main():
    from subjects.adf_data.census import census_data
    from subjects.adf_data.credit import credit_data
    from subjects.adf_data.bank import bank_data
    from subjects.adf_data.compas import compas_data
    import pandas as pd
    from configs import columns, get_groups, labeled_df, int_to_cat_labels_map
    vis_technique = ["Correlation"]
    datasets = [("census", "sex",9), ("census", "race",8), ("credit", "sex",9), ("bank","age",1), ("compas","sex",1), ("compas","race",3)]




    picked_vis_technique = st.radio("Visualization technique:", vis_technique, horizontal = True)
    picked_dataset = st.radio("Dataset:", datasets, horizontal = True)
    sensitive_param = picked_dataset[2]
    sensitive_name = picked_dataset[1]
    
    if picked_vis_technique == "Correlation":
        from sklearn.covariance import EmpiricalCovariance
        from sklearn.preprocessing import StandardScaler
        data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}
        X, Y, input_shape, nb_classes = data[picked_dataset[0]]()
        Y = np.argmax(Y, axis=1)
        # For this one I actually want to visualize the labels as well 
        all_data = pd.concat([pd.DataFrame(X),pd.DataFrame(Y)], axis=1) 
        X_transformed = StandardScaler().fit_transform(all_data) # This turns the "covariance" into correlation
        
        cov = EmpiricalCovariance().fit(X_transformed)
        
        # bar_data = {"Column number": nb_classes, "Correlation": cov.covariance_[sensitive_param-1]}
        sensitive_feature = ["blue"]*len(cov.covariance_[sensitive_param-1])
        sensitive_feature[sensitive_param-1] = "red"
        sensitive_feature[len(sensitive_feature)-1] = "green"
        X_transformed = pd.DataFrame(X_transformed)
        X = pd.DataFrame(X)
        X = pd.concat([X, pd.DataFrame(Y)], axis=1)
        X.columns = columns[picked_dataset[0]]
        X_transformed.columns = columns[picked_dataset[0]]
        bar_data = {"Column number": columns[picked_dataset[0]], "Correlation": cov.covariance_[sensitive_param-1], "sensitive_feature": [False]*len(columns[picked_dataset[0]])}
        bar_data["sensitive_feature"][picked_dataset[2]-1] = True
        bar_data = pd.DataFrame(bar_data)
        st.write("Note: Many of the data are categorical, making correlation a bad metric. Click on the bar to see more detailed labelled bar/histogram graphs.")
        fig = plt.bar(bar_data, x="Column number", y="Correlation", color="sensitive_feature", category_orders={"Column number": columns[picked_dataset[0]]},color_discrete_map={False:'blue', True:'red'}, title=f"Correlation of each column with {sensitive_name}")
        bar_graph = plotly_events(fig)
        if bar_graph != []:
            # n_bins = st.slider("Number of desired bins", 0, 100, 15)
            X_labeled = labeled_df(X, picked_dataset[0])
            cat_labels = int_to_cat_labels_map(picked_dataset[0])
            group_0, group_1 = get_groups(picked_dataset[0], sensitive_name, get_name=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=X_labeled[bar_graph[0]["x"]][(X_labeled[sensitive_name]==group_0)],histnorm="", histfunc="count", name=f'{group_0}'))
            fig2.add_trace(go.Histogram(x=X_labeled[bar_graph[0]["x"]][(X_labeled[sensitive_name]==group_1)],histnorm="", histfunc="count", name=f'{group_1}'))
            fig2.update_layout(
                title_text=f"Histogram of {sensitive_name} v.s. {bar_graph[0]['x']}", # title of plot
                xaxis_title_text=f"{bar_graph[0]['x']}", # xaxis label
                yaxis_title_text='Count', # yaxis label
                bargap=0.2, # gap between bars of adjacent location coordinates
                bargroupgap=0.1 # gap between bars of the same location coordinates
            )


            feature_sensitive_percentage = X_labeled[bar_graph[0]["x"]][(X_labeled[sensitive_name]==group_0)].value_counts()/(X_labeled[bar_graph[0]["x"]].value_counts())*100
            feature_sensitive_percentage = feature_sensitive_percentage.rename(group_0)
            fig3 = plt.bar(feature_sensitive_percentage, title=f"Histogram of % {group_0} v.s. {bar_graph[0]['x']} scaled")
            fig3.update_layout(
                title_text=f"Histogram of % {group_0} v.s. {bar_graph[0]['x']} scaled", # title of plot
                xaxis_title_text=f"{bar_graph[0]['x']}", # xaxis label
                yaxis_title_text='Percentage', # yaxis label
                bargap=0.2, # gap between bars of adjacent location coordinates
                bargroupgap=0.1 # gap between bars of the same location coordinates
            )

            if bar_graph[0]["x"] in cat_labels.keys():
                fig2.update_xaxes(categoryorder='array', categoryarray=list(cat_labels[bar_graph[0]["x"]].values()))
                fig3.update_xaxes(categoryorder='array', categoryarray=list(cat_labels[bar_graph[0]["x"]].values()))
            else:
                fig2.update_xaxes(categoryorder='category ascending')
                fig3.update_xaxes(categoryorder='category ascending')
            
            
            hist_graph = plotly_events(fig2)
            hist_graph = plotly_events(fig3)
            st.write(f"The average percentage for {group_0} is {X_labeled[X_labeled[sensitive_name]==group_0][sensitive_name].count()/X_labeled[sensitive_name].count()}")

        # fig2 = plt.bar()
    
    

