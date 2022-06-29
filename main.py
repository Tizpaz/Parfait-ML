import subprocess

subprocess.call(['./mutation_testing.sh'])


import pandas as pd

classifiers = ["LR", "RF", "SV", "DT"]
test = [("census", "gender"), ("census", "race"), ("credit", "gender"), ("bank", "age"), ("compas", "gender"), ("compas", "race")]

inps = []

for t in test:
    for c in classifiers:
        df = pd.read_csv("./Dataset" + "/" + f"{c}_{t[0]}_{t[1]}_mutation.csv")
        inps.append(df[df['score'] == df['score'].min()].iloc[0].inp)

weight_command = ['./weight_exhaustion_testing.sh'] + inps

subprocess.call(weight_command)