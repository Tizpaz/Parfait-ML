python3 weight_exhaustion.py --dataset=census --algorithm=LogisticRegressionMitigation --sensitive_index=9 --time_out=1440 --output=LR_census_gender_masking
python3 weight_exhaustion.py --dataset=census --algorithm=LogisticRegressionMitigation --sensitive_index=8 --time_out=1440 --output=LR_census_race_masking
python3 weight_exhaustion.py --dataset=credit --algorithm=LogisticRegressionMitigation --sensitive_index=9 --mtime_out=1440 --output=LR_credit_gender_masking
python3 weight_exhaustion.py --dataset=bank --algorithm=LogisticRegressionMitigation --sensitive_index=1 --time_out=1440 --output=LR_bank_age_masking
python3 weight_exhaustion.py --dataset=compas --algorithm=LogisticRegressionMitigation --sensitive_index=1 --time_out=1440 --output=LR_compas_gender_masking
python3 weight_exhaustion.py --dataset=compas --algorithm=LogisticRegressionMitigation --sensitive_index=3 --time_out=1440 --output=LR_compas_race_masking

python3 weight_exhaustion.py --dataset=census --algorithm=TreeRegressorMitigation --sensitive_index=9 --time_out=1440 --output=RF_census_gender_masking
python3 weight_exhaustion.py --dataset=census --algorithm=TreeRegressorMitigation --sensitive_index=8 --time_out=1440 --output=RF_census_race_masking
python3 weight_exhaustion.py --dataset=credit --algorithm=TreeRegressorMitigation --sensitive_index=9 --time_out=1440 --output=RF_credit_gender_masking
python3 weight_exhaustion.py --dataset=bank --algorithm=TreeRegressorMitigation --sensitive_index=1 -time_out=1440 --output=RF_bank_age_masking
python3 weight_exhaustion.py --dataset=compas --algorithm=TreeRegressorMitigation --sensitive_index=1 --time_out=1440 --output=RF_compas_gender_masking
python3 weight_exhaustion.py --dataset=compas --algorithm=TreeRegressorMitigation --sensitive_index=3 --time_out=1440 --output=RF_compas_race_masking

python3 weight_exhaustion.py --dataset=census --algorithm=SVM_Mitigation --sensitive_index=9 -time_out=1440 --output=SV_census_gender_masking
python3 weight_exhaustion.py --dataset=census --algorithm=SVM_Mitigation --sensitive_index=8 -time_out=1440 --output=SV_census_race_masking
python3 weight_exhaustion.py --dataset=credit --algorithm=SVM_Mitigation --sensitive_index=9 --mtime_out=1440 --output=SV_credit_gender_masking
python3 weight_exhaustion.py --dataset=bank --algorithm=SVM_Mitigation --sensitive_index=1 -time_out=1440 --output=SV_bank_age_masking
python3 weight_exhaustion.py --dataset=compas --algorithm=SVM_Mitigation --sensitive_index=1 --time_out=1440 --output=SV_compas_gender_masking
python3 weight_exhaustion.py --dataset=compas --algorithm=SVM_Mitigation --sensitive_index=3 --time_out=1440 --output=SV_compas_race_masking

python3 weight_exhaustion.py --dataset=census --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=9 --time_out=1440 --outpt=DT_census_gender_masking
python3 weight_exhaustion.py --dataset=census --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=8 --time_out=1440 --outpt=DT_census_race_masking
python3 weight_exhaustion.py --dataset=credit --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=9 --mtime_out=1440 --outpt=DT_credit_gender_masking
python3 weight_exhaustion.py --dataset=bank --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=1 --time_out=1440 --outpt=DT_bank_age_masking
python3 weight_exhaustion.py --dataset=compas --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=1 --time_out=1440 --outpt=DT_compas_gender_masking
python3 weight_exhaustion.py --dataset=compas --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=3 --time_out=1440 --outpt=DT_compas_race_masking