columns = {"bank":["age","job","martial","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","label"],
            "census": ["age","workclass","fnlwgt","education-num","martial-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","label"],
            "compas": ["sex","age","race","d","e","f","g","h","i","j","k","l","label"],
            "credit": ["checking-account-status","duration-months","credit-history","purpose","credit-amount","saving-account-status","employment-status","disposable-income-percent","sex","other-debts","current-residence-time","property-type","age","installment-plans","housing","num-credit-lines","job-type","num-dependents","telephone","foreigner","label"]
    }

def get_groups(dataset, sensitive_name):
        if dataset == "census" and (sensitive_name == "sex" or sensitive_name == "gender"):
                sensitive_param = 9
                group_0 = 0  #female
                group_1 = 1  #male
        if dataset == "census" and sensitive_name == "race":
                sensitive_param = 8
                group_0 = 0 # white
                group_1 = 4 # black
        if dataset == "credit" and (sensitive_name == "sex" or sensitive_name == "gender"):
                sensitive_param = 9
                group_0 = 0  # male
                group_1 = 1  # female
        if dataset == "bank" and sensitive_name == "age":  # with 3,5: 0.89; with 2,5: 0.84; with 4,5: 0.05; with 3,4: 0.6
                group_0 = 3
                group_1 = 5
                sensitive_param = 1
        if dataset == "compas" and (sensitive_name == "sex" or sensitive_name == "gender"):  # sex
                group_0 = 0 # male
                group_1 = 1 # female
                sensitive_param = 1
        if dataset == "compas" and sensitive_name == "age":  # age
                group_0 = 0 # under 25
                group_1 = 2 # greater than 45
                sensitive_param = 2
        if dataset == "compas" and sensitive_name == "race":  # race
                group_0 = 0 # non-Caucasian
                group_1 = 1 # Caucasian
                sensitive_param = 3
        return group_0, group_1