#!/bin/sh
python3 weight_exhaustion.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9 --time_out=1440 --output=LR_census_gender_masking --inp=$1
python3 weight_exhaustion.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --time_out=1440 --output=LR_census_race_masking --inp=$2
python3 weight_exhaustion.py --dataset=credit --algorithm=LogisticRegression --sensitive_index=9 --mtime_out=1440 --output=LR_credit_gender_masking --inp=$3
python3 weight_exhaustion.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1 --time_out=1440 --output=LR_bank_age_masking --inp=$4
python3 weight_exhaustion.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=1 --time_out=1440 --output=LR_compas_gender_masking --inp=$5
python3 weight_exhaustion.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=3 --time_out=1440 --output=LR_compas_race_masking --inp=$6

python3 weight_exhaustion.py --dataset=census --algorithm=TreeRegressor --sensitive_index=9 --time_out=1440 --output=RF_census_gender_masking --inp=$7
python3 weight_exhaustion.py --dataset=census --algorithm=TreeRegressor --sensitive_index=8 --time_out=1440 --output=RF_census_race_masking --inp=$8
python3 weight_exhaustion.py --dataset=credit --algorithm=TreeRegressor --sensitive_index=9 --time_out=1440 --output=RF_credit_gender_masking --inp=$9
python3 weight_exhaustion.py --dataset=bank --algorithm=TreeRegressor --sensitive_index=1 --time_out=1440 --output=RF_bank_age_masking --inp=${10}
python3 weight_exhaustion.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=1 --time_out=1440 --output=RF_compas_gender_masking --inp=${11}
python3 weight_exhaustion.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=3 --time_out=1440 --output=RF_compas_race_masking --inp=${12}

python3 weight_exhaustion.py --dataset=census --algorithm=SVM --sensitive_index=9 -time_out=1440 --output=SV_census_gender_masking --inp=${13}
python3 weight_exhaustion.py --dataset=census --algorithm=SVM --sensitive_index=8 -time_out=1440 --output=SV_census_race_masking --inp=${14}
python3 weight_exhaustion.py --dataset=credit --algorithm=SVM --sensitive_index=9 --time_out=1440 --output=SV_credit_gender_masking --inp=${15}
python3 weight_exhaustion.py --dataset=bank --algorithm=SVM --sensitive_index=1 --time_out=1440 --output=SV_bank_age_masking --inp=${16}
python3 weight_exhaustion.py --dataset=compas --algorithm=SVM --sensitive_index=1 --time_out=1440 --output=SV_compas_gender_masking --inp=${17}
python3 weight_exhaustion.py --dataset=compas --algorithm=SVM --sensitive_index=3 --time_out=1440 --output=SV_compas_race_masking --inp=${18}

python3 weight_exhaustion.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9 --time_out=1440 --output=DT_census_gender_masking --inp=${19}
python3 weight_exhaustion.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=8 --time_out=1440 --output=DT_census_race_masking --inp=${20}
python3 weight_exhaustion.py --dataset=credit --algorithm=Decision_Tree_Classifier --sensitive_index=9 --time_out=1440 --output=DT_credit_gender_masking --inp=${21}
python3 weight_exhaustion.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1 --time_out=1440 --output=DT_bank_age_masking --inp=${22}
python3 weight_exhaustion.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=1 --time_out=1440 --output=DT_compas_gender_masking --inp=${23}
python3 weight_exhaustion.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=3 --time_out=1440 --output=DT_compas_race_masking --inp=${24}