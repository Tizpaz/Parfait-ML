assert(False) #I'm just trying to not test this for now
import pandas as pd
import sys
def getMostAccurateConfig(model,dataset,sensitive_feature,algo):
    df = pd.read_csv("./Dataset" + "/" + f"{model}_{dataset}_{sensitive_feature}_{algo}.csv")
    return df[df['score'] == df['score'].max()].iloc[0].inp
import subprocess
models = [("LR", "LogisticRegression"),("RF","TreeRegressor"),("SV","SVM"),("DT","Decision_Tree_Classifier")]
datasets = [("census", "gender",9), ("census", "race",8), ("credit", "gender",9), ("bank","age",1), ("compas","gender",1), ("compas","race",3)]
timeout = sys.argv[1]
save_model = sys.argv[2]
for m in models:
    for d in datasets:
        most_accurate = getMostAccurateConfig(m[0],d[0],d[1],"mutation")
        running_process = ["python3", "weight_exhaustion.py", f"--dataset={d[0]}", f"--algorithm={m[1]}", f"--sensitive_index={d[2]}", f"--time_out={timeout}",\
            f"--output={m[0]}_{d[0]}_{d[1]}_masking", f"--inp={most_accurate}", f"--save_model={save_model}"]
        print(running_process)
        subprocess.run(running_process)