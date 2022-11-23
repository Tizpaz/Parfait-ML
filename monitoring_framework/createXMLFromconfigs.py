import configs
import os
def populateInput(col_name, type, values):
    inputCapsule= f"""        <input>
            <name>{col_name}</name>
            <type>{type}</type>
{populateValues(values, type == "categorical")}
        </input>"""
    return inputCapsule

def populateValues(values, cat):
    if cat:
        valueCapsule = """            <values>"""
        for val in values:
            valueCapsule+= f"""\n                <value>{val.replace("<", "&lt;").replace("&","&amp;")}</value>"""
        
        valueCapsule+= """\n            </values>"""
    else:
        valueCapsule = """            <bounds>"""
        valueCapsule+= f"""\n                <lowerbound>{values[0]}</lowerbound>"""
        valueCapsule+= f"""\n                <upperbound>{values[1]}</upperbound>"""
        valueCapsule+= """\n            </bounds>"""
    return valueCapsule

def populateSettings(col_names, types, arrValues):
    settingsCapsule = f"""<settings>
    <name>Software</name>
    <seed>42</seed>
    <max_samples>1000</max_samples>
    <min_samples>3</min_samples>
    <inputs>"""
    for i in range(len(col_names)):
        settingsCapsule += f"""\n{populateInput(col_names[i], types[i], arrValues[i])}"""
    settingsCapsule += '''\n    </inputs>
</settings>'''
    return settingsCapsule

def getBoundsFromData():
    import numpy as np
    import math
    import matplotlib.pyplot as mplt
    import sys
    sys.path.append("./")
    sys.path.append("../")
    from subjects.adf_data.census import census_data
    from subjects.adf_data.credit import credit_data
    from subjects.adf_data.bank import bank_data
    from subjects.adf_data.compas import compas_data
    import pandas as pd
    from configs import categorical_features, columns, get_groups, labeled_df, int_to_cat_labels_map
    for picked_dataset in columns.keys():
        data = {"census":census_data, "credit":credit_data, "bank":bank_data, "compas": compas_data}
        X, Y, input_shape, nb_classes = data[picked_dataset]()
        X = pd.DataFrame(X)
        Y = np.argmax(Y, axis=1)
        X = pd.concat([X, pd.DataFrame(Y)], axis=1)
        X.columns = columns[picked_dataset]
        print(picked_dataset)
        for i in range(len(columns[picked_dataset])):
            if i not in categorical_features[picked_dataset]:
                print(f"{int(X[columns[picked_dataset][i]].min())},{int(X[columns[picked_dataset][i]].max())}")


# getBoundsFromData()


for dataset in configs.columns.keys():
    col_names = configs.columns[dataset]
    types = [("categorical" if i in configs.categorical_features[dataset] else "continuousInt") for i in range(len(col_names))]
    arrValues = []
    cat_values = list(configs.cat_to_int_map(dataset).values())
    num_values = configs.numerical_bounds[dataset]
    c1 = 0
    c2 = 0
    for i in range(len(configs.columns[dataset])):

        if i in configs.categorical_features[dataset]:
            arrValues.append(list(cat_values[c1].keys()))
            c1 += 1
        else:
            arrValues.append(num_values[c2].split(","))
            c2 += 1
    file_path = os.path.realpath(os.path.dirname(__file__))
    f= open(f"{file_path}/Themis/Themis2/settings_{dataset}.xml", 'w')
    f.write(populateSettings(col_names[:-1], types[:-1], arrValues[:-1]))
    f.close()

