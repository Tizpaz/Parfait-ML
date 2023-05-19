#!/bin/sh
python3 main_mutation.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9 --time_out=$1 --output=LR_census_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --time_out=$1 --output=LR_census_race_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=credit --algorithm=LogisticRegression --sensitive_index=9 --time_out=$1 --output=LR_credit_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1 --time_out=$1 --output=LR_bank_age_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=1 --time_out=$1 --output=LR_compas_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=3 --time_out=$1 --output=LR_compas_race_mutation --save_model=$2 --max_iter=$3

python3 main_mutation.py --dataset=census --algorithm=TreeRegressor --sensitive_index=9 --time_out=$1 --output=RF_census_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=census --algorithm=TreeRegressor --sensitive_index=8 --time_out=$1 --output=RF_census_race_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=credit --algorithm=TreeRegressor --sensitive_index=9 --time_out=$1 --output=RF_credit_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=bank --algorithm=TreeRegressor --sensitive_index=1 --time_out=$1 --output=RF_bank_age_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=1 --time_out=$1 --output=RF_compas_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=3 --time_out=$1 --output=RF_compas_race_mutation --save_model=$2 --max_iter=$3

python3 main_mutation.py --dataset=census --algorithm=SVM --sensitive_index=9 --time_out=$1 --output=SV_census_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=census --algorithm=SVM --sensitive_index=8 --time_out=$1 --output=SV_census_race_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=credit --algorithm=SVM --sensitive_index=9 --time_out=$1 --output=SV_credit_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=bank --algorithm=SVM --sensitive_index=1 --time_out=$1 --output=SV_bank_age_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=compas --algorithm=SVM --sensitive_index=1 --time_out=$1 --output=SV_compas_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=compas --algorithm=SVM --sensitive_index=3 --time_out=$1 --output=SV_compas_race_mutation --save_model=$2 --max_iter=$3

python3 main_mutation.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9 --time_out=$1 --output=DT_census_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=8 --time_out=$1 --output=DT_census_race_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=credit --algorithm=Decision_Tree_Classifier --sensitive_index=9 --time_out=$1 --output=DT_credit_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1 --time_out=$1 --output=DT_bank_age_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=1 --time_out=$1 --output=DT_compas_gender_mutation --save_model=$2 --max_iter=$3
python3 main_mutation.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=3 --time_out=$1 --output=DT_compas_race_mutation --save_model=$2 --max_iter=$3
