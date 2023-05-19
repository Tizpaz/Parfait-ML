import copy

from pandas import value_counts


columns = {
    "bank":["age","job","martial","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","label"],
    "census": ["age","workclass","fnlwgt","education","martial-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","label"],
    "compas": ["sex","age","race","d","e","f","g","h","i","j","k","l","label"],
    "credit": ["checking-account-status","duration-months","credit-history","purpose","credit-amount","saving-account-status","employment-status","disposable-income-percent","sex","other-debts","current-residence-time","property-type","age","installment-plans","housing","num-credit-lines","job-type","num-dependents","telephone","foreigner","label"]
}
categorical_features = {
    "bank": [0,1,2,3,4,6,7,8,9,10,15,16],
    "census": [0,1,3,4,5,6,7,8,12,13],
    "compas": [0,1,2,12], # Incomplete: when in doubt we just assumed numerical
    "credit": [0,2,3,5,6,8,9,11,13,14,16,18,19,20]
}

numerical_bounds = {
    "bank": '''-20,179
0,99
1,63
0,1
0,1'''.split('\n'),
    "census": '''0,74
0,99
0,43
1,99'''.split('\n'),
    "compas": '''0,20
1,10
0,38
0,1
0,1
0,1
1,10
1,10
0,38'''.split('\n'),
    "credit": '''4,72
250,18424
1,4
1,4
19,75
1,4
1,2'''.split('\n')
}

categorical_features_names = {
    # https://archive.ics.uci.edu/ml/datasets/bank+marketing for the categorical ones)) -- data also matches bank-full.csv in our dataset
    "bank": '''young-children(0-10)=0,teens(10-20)=1,20s=2,30s=3,40s=4,50s=5,60s=6,70s=7,80s=8,90s=9
admin=0,blue-collar=1,entrepreneur=2,housemaid=3,management=4,retired=5,self-employed=6,services=7,student=8,technician=9,unemployed=10,unknown=11
divorced=0,married=1,single=2,unknown=3
basic-4y=0,basic-6y=1,basic.9y=2,high-school=3,illiterate=4,professional-course=5,university-degree=6,unknown=7
no=0,yes=1,unknown=2
no=0,yes=1,unknown=2
no=0,yes=1,unknown=2
cellular=0,telephone=1,unknown=2
1=1,2=2,3=3,4=4,5=5,6=6,7=7,8=8,9=9,10=10,11=11,12=12,13=13,14=14,15=15,16=16,17=17,18=18,19=19,20=20,21=21,22=22,23=23,24=24,25=25,26=26,27=27,28=28,29=29,30=30,31=31
jan=0,feb=1,mar=2,apr=3,may=4,jun=5,jul=6,aug=7,sep=8,oct=9,nov=10,dec=11
failure=0,nonexistent=1,success=2,other=3
subscribed=0,not-subscribed=1'''.split('\n'),
    # (https://archive.ics.uci.edu/ml/datasets/census+income for the categorical ones)
    "census": '''young-children(0-10)=0,teens(10-20)=1,20s=2,30s=3,40s=4,50s=5,60s=6,70s=7,80s=8,90s=9
Private=0,Self-emp-not-inc=1,Self-emp-inc=2,Federal-gov=3,Local-gov=4,State-gov=5,Without-pay=6,Never-worked=7,unknown=100
Bachelors=0,Some-college=1,11th=2,HS-grad=3,Prof-school=4,Assoc-acdm=5,Assoc-voc=6,9th=7,7th-8th=8,12th=9,Masters=10,1st-4th=11,10th=12,Doctorate=13,5th-6th=14,Preschool=15
Married-civ-spouse=0,Divorced=1,Never-married=2,Separated=3,Widowed=4,Married-spouse-absent=5,Married-AF-spouse=6
Tech-support=0,Craft-repair=1,Other-service=2,Sales=3,Exec-managerial=4,Prof-specialty=5,Handlers-cleaners=6,Machine-op-inspct=7,Adm-clerical=8,Farming-fishing=9,Transport-moving=10,Priv-house-serv=11,Protective-serv=12,Armed-Forces=13,unknown=100
Wife=0,Own-child=1,Husband=2,Not-in-family=3,Other-relative=4,Unmarried=5
White=0,Asian-Pac-Islander=1,Amer-Indian-Eskimo=2,Other=3,Black=4
Female=0,Male=1
United-States=0,Cambodia=1,England=2,Puerto-Rico=3,Canada=4,Germany=5,Outlying-US(Guam-USVI-etc)=6,India=7,Japan=8,Greece=9,South-Africa=10,China=11,Cuba=12,Iran=13,Honduras=14,Philippines=15,Italy=16,Poland=17,Jamaica=18,Vietnam=19,Mexico=20,Portugal=21,Ireland=22,France=23,Dominican-Republic=24,Laos=25,Ecuador=26,Taiwan=27,Haiti=28,Columbia=29,Hungary=30,Guatemala=31,Nicaragua=32,Scotland=33,Thailand=34,Yugoslavia=35,El-Salvador=36,Trinadad&Tobago=37,Peru=38,Hong-Kong=39,Holand-Netherlands=40,unknown=100
less-than-50K=0,more-than-50K=1'''.split('\n'),
    # (Follows sensitive attribute code in https://github.com/tizpaz/parfait-ml)
    # (Also gussed based on: https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb)
    "compas": '''male=0,female=1
0-25=0,25-45=1,45+=2
non-caucasian=0,caucasian=1
low-recid-risk=0,high-recid-risk=1'''.split('\n'),
    # (https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data) index start at 1 if first item is Axx1, index starts at 0 if first item is AXX0)
    "credit": '''owe-money=1,0-200DM=2,200+DM=3,no-account=4
all-credit-paid-on-time=0,cur-ank-all-credit-paid-on-time=1,existing-credit-paid-on-time=2,day-in-paying-credit=3,critical-accounts=4
new-car=0,used-car=1,furniture=2,radio-television=3,domestic-appliance=4,repairs=5,education=6,vacation=7,retraining=8,buisness=9,others=10
<100DM=1,100-500DM=2,500-1000DM=3,1000+DM=4,unknown-nosavings=5
unemployed=1,<1yr=2,1-4yr=3,4-7yr=4,7+yr=5
male=0,female=1
none=1,co-applicant=2,guarantor=3
real-estate=1,insurance=2,car=3,unknown=4
bank=1,stores=2,none=3
rent=1,own=2,free=3
unemployed(non-resident)=1,unemployed(resident)=2,low-skill-employed=3,high-skill-employed=4
none=1,yes=2
yes=1,no=2
not-risky=0,risky=1'''.split('\n')
}



def labeled_df(org_df, dataset):
    labeled_df = copy.deepcopy(org_df)
    for j, names in enumerate(categorical_features_names[dataset]):
        values = names.split(',')
        values = dict([(x.split('=')[1], x.split('=')[0]) for x in values])
        labeled_df.iloc[:,categorical_features[dataset][j]] = list(map(lambda x: values[str(int(x))], labeled_df.iloc[:,categorical_features[dataset][j]]))
    return labeled_df
def int_to_cat_labels_map(dataset):
    output = {}
    for j, names in enumerate(categorical_features_names[dataset]):
        values = names.split(',')
        output[columns[dataset][categorical_features[dataset][j]]] = dict([(x.split('=')[1], x.split('=')[0]) for x in values])
    return output
def cat_to_int_map(dataset):
    output = {}
    for j, names in enumerate(categorical_features_names[dataset]):
        values = names.split(',')
        output[columns[dataset][categorical_features[dataset][j]]] = dict([(x.split('=')[0], x.split('=')[1]) for x in values])
    return output
def get_groups(dataset, sensitive_name, get_name=False):
    if get_name:
        if dataset == "census" and (sensitive_name == "sex" or sensitive_name == "gender"):
            sensitive_param = 9
            group_0 = "Female"
            group_1 = "Male"
        if dataset == "census" and sensitive_name == "race":
            sensitive_param = 8
            group_0 = "White"
            group_1 = "Black"
        if dataset == "credit" and (sensitive_name == "sex" or sensitive_name == "gender"):
            sensitive_param = 9
            group_0 = "male"
            group_1 = "female"
        if dataset == "bank" and sensitive_name == "age":  # with 3,5: 0.89; with 2,5: 0.84; with 4,5: 0.05; with 3,4: 0.6
            group_0 = "30s"
            group_1 = "50s"
            sensitive_param = 1
        if dataset == "compas" and (sensitive_name == "sex" or sensitive_name == "gender"):  # sex
            group_0 = "male"
            group_1 = "female"
            sensitive_param = 1
        if dataset == "compas" and sensitive_name == "age":  # age
            group_0 = "0-25"
            group_1 = "45+"
            sensitive_param = 2
        if dataset == "compas" and sensitive_name == "race":  # race
            group_0 = "non-caucasian"
            group_1 = "caucasian"
            sensitive_param = 3
    else:
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