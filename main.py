import subprocess
timeout=90
max_iter=50
save_model = True
subprocess.call(['./m_mutation_testing.sh', f"{timeout}", f"{save_model}", f"{max_iter}"])


import pandas as pd

classifiers = ["LR", "RF", "SV", "DT"]
test = [("census", "gender"), ("census", "race"), ("credit", "gender"), ("bank", "age"), ("compas", "gender"), ("compas", "race")]

inps = []

for t in test:
    for c in classifiers:
        df = pd.read_csv("./Dataset" + "/" + f"{c}_{t[0]}_{t[1]}_mutation.csv")
        inps.append(df[df['score'] == df['score'].min()].iloc[0].inp)

weight_command = ['./m_weight_exhaustion_testing.py', f"{timeout}", f"{save_model}"]

subprocess.call(weight_command)