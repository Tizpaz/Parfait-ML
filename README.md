# ML-code-fairness
### Implementations for detecting and understanding biases in ML libraries
### Requirements
Python 3.6, fairlearn (e.g., pip install fairlearn), standard python packages such as Numpy, Scikit-learn, pydotplus.  
Grphviz installed in your system.

Run the random search for adult census dataset with gender as sensitive feature:
```
python main_random.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000  2>&1 | tee LogisticRegression_census_gender_random_output.txt &
python main_random.py --dataset=census --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000  2>&1 | tee TreeRegressor_census_gender_random_output.txt &
python main_random.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_census_gender_random_output.txt &
python main_random.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_gender_random_output.txt &
python main_random.py --dataset=census --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_census_gender_random_output.txt &
```
This will result into two files: .csv file shows inputs and outcome of training; .txt file writes extra details (not essential part of the analysis).    

Run the other commands with other datasets as the following:   
```
python main_mutation.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000 2>&1 | tee LogisticRegression_census_gender_mutation_output.txt &
python main_mutation.py --dataset=census --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000 2>&1 | tee TreeRegressor_census_gender_mutation_output.txt &
python main_mutation.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_census_gender_mutation_output.txt &
python main_mutation.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_gender_mutation_output.txt &
python main_mutation.py --dataset=census --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_census_gender_mutation_output.txt &
```
```
python main_coverage.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000  2>&1 | tee LogisticRegression_census_gender_coverage_output.txt &
python main_coverage.py --dataset=census --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000 2>&1 | tee TreeRegressor_census_gender_coverage_output.txt &
python main_coverage.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_census_gender_coverage_output.txt &
python main_coverage.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_gender_coverage_output.txt &
python main_coverage.py --dataset=census --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_census_gender_coverage_output.txt &
```
```
python main_random.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --max_iter=100000 2>&1 | tee LogisticRegression_census_race_random_output.txt &
python main_random.py --dataset=census --algorithm=TreeRegressor --sensitive_index=8 --max_iter=100000  2>&1 | tee TreeRegressor_census_race_random_output.txt &
python main_random.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=8 --max_iter=100000  2>&1 | tee Decision_Tree_Classifier_census_race_random_output.txt &
python main_random.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=8 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_race_random_output.txt &
python main_random.py --dataset=census --algorithm=SVM --sensitive_index=8 --max_iter=100000 2>&1 | tee SVM_census_race_random_output.txt &
```
```
python main_mutation.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --max_iter=100000 2>&1 | tee LogisticRegression_census_race_mutation_output.txt &
python main_mutation.py --dataset=census --algorithm=TreeRegressor --sensitive_index=8 --max_iter=100000 2>&1 | tee TreeRegressor_census_race_mutation_output.txt &
python main_mutation.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=8 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_census_race_mutation_output.txt &
python main_mutation.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=8 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_race_mutation_output.txt &
python main_mutation.py --dataset=census --algorithm=SVM --sensitive_index=8 --max_iter=100000 2>&1 | tee SVM_census_race_mutation_output.txt &
```
```
python main_coverage.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --max_iter=100000  2>&1 | tee LogisticRegression_census_race_coverage_output.txt &
python main_coverage.py --dataset=census --algorithm=TreeRegressor --sensitive_index=8 --max_iter=100000 2>&1 | tee TreeRegressor_census_race_coverage_output.txt &
python main_coverage.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=8 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_census_race_coverage_output.txt &
python main_coverage.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=8 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_race_coverage_output.txt &
python main_coverage.py --dataset=census --algorithm=SVM --sensitive_index=8 --max_iter=100000 2>&1 | tee SVM_census_race_coverage_output.txt &
```
```
python main_random.py --dataset=credit --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000 2>&1 | tee LogisticRegression_credit_gender_random_output.txt &
python main_random.py --dataset=credit --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000 2>&1 | tee TreeRegressor_credit_gender_random_output.txt &
python main_random.py --dataset=credit --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_credit_random_random_output.txt &
python main_random.py --dataset=credit --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_credit_gender_random_output.txt &
python main_random.py --dataset=credit --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_credit_gender_random_output.txt &
```
```
python main_mutation.py --dataset=credit --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000 2>&1 | tee LogisticRegression_credit_gender_mutation_output.txt &
python main_mutation.py --dataset=credit --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000 2>&1 | tee TreeRegressor_credit_gender_mutation_output.txt &
python main_mutation.py --dataset=credit --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_credit_gender_mutation_output.txt &
python main_mutation.py --dataset=credit --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_credit_gender_mutation_output.txt &
python main_mutation.py --dataset=credit --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_credit_gender_mutation_output.txt &
```
```
python main_coverage.py --dataset=credit --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000 2>&1 | tee LogisticRegression_credit_gender_coverage_output.txt &
python main_coverage.py --dataset=credit --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000 2>&1 | tee TreeRegressor_credit_gender_coverage_output.txt &
python main_coverage.py --dataset=credit --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_credit_gender_coverage_output.txt &
python main_coverage.py --dataset=credit --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_credit_gender_coverage_output.txt &
python main_coverage.py --dataset=credit --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_credit_gender_coverage_output.txt &
```
```
python main_random.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_bank_random_output.txt &
python main_random.py --dataset=bank --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_bank_random_output.txt &
python main_random.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_bank_random_output.txt &
python main_random.py --dataset=bank --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_bank_random_output.txt &
python main_random.py --dataset=bank --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_bank_random_output.txt &
```
```
python main_mutation.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_bank_mutation_output.txt &
python main_mutation.py --dataset=bank --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_bank_mutation_output.txt &
python main_mutation.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_bank_mutation_output.txt &
python main_mutation.py --dataset=bank --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_bank_mutation_output.txt &
python main_mutation.py --dataset=bank --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_bank_mutation_output.txt &
```
```
python main_coverage.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_bank_coverage_output.txt &
python main_coverage.py --dataset=bank --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_bank_coverage_output.txt &
python main_coverage.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_bank_coverage_output.txt &
python main_coverage.py --dataset=bank --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_bank_coverage_output.txt &
python main_coverage.py --dataset=bank --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_bank_coverage_output.txt &
```
```
python main_random.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_compas_gender_random_output.txt &
python main_random.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_compas_gender_random_output.txt &
python main_random.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_gender_random_output.txt &
python main_random.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_gender_random_output.txt &
python main_random.py --dataset=compas --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_compas_gender_random_output.txt &
```
```
python main_mutation.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_compas_gender_mutation_output.txt &
python main_mutation.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_compas_gender_mutation_output.txt &
python main_mutation.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_gender_mutation_output.txt &
python main_mutation.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_gender_mutation_output.txt &
python main_mutation.py --dataset=compas --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_compas_gender_mutation_output.txt &
```
```
python main_coverage.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_compas_gender_coverage_output.txt &
python main_coverage.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_compas_gender_coverage_output.txt &
python main_coverage.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_gender_coverage_output.txt &
python main_coverage.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_gender_coverage_output.txt &
python main_coverage.py --dataset=compas --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_compas_gender_coverage_output.txt &
```
```
python main_random.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=3 --max_iter=100000  2>&1 | tee LogisticRegression_compas_race_random_output.txt &
python main_random.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=3 --max_iter=100000 2>&1 | tee TreeRegressor_compas_race_random_output.txt &
python main_random.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=3 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_race_random_output.txt &
python main_random.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=3 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_race_random_output.txt &
python main_random.py --dataset=compas --algorithm=SVM --sensitive_index=3 --max_iter=100000 2>&1 | tee SVM_compas_race_random_output.txt &
```
```
python main_mutation.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=3 --max_iter=100000 2>&1 | tee LogisticRegression_compas_race_mutation_output.txt &
python main_mutation.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=3 --max_iter=100000 2>&1 | tee TreeRegressor_compas_race_mutation_output.txt &
python main_mutation.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=3 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_race_mutation_output.txt &
python main_mutation.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=3 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_race_mutation_output.txt &
python main_mutation.py --dataset=compas --algorithm=SVM --sensitive_index=3 --max_iter=100000 2>&1 | tee SVM_compas_race_mutation_output.txt &
```
```
python main_coverage.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=3 --max_iter=100000 2>&1 | tee LogisticRegression_compas_race_coverage_output.txt &
python main_coverage.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=3 --max_iter=100000 2>&1 | tee TreeRegressor_compas_race_coverage_output.txt &
python main_coverage.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=3 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_race_coverage_output.txt &
python main_coverage.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=3 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_race_coverage_output.txt &
python main_coverage.py --dataset=compas --algorithm=SVM --sensitive_index=3 --max_iter=100000 2>&1 | tee SVM_compas_race_coverage_output.txt &
```
Here, we combine our tool with ExponentiatedGradient approach from fairlearn library:
```
python main_mutation.py --dataset=census --algorithm=LogisticRegressionMitigation --sensitive_index=9 --max_iter=150
python main_mutation.py --dataset=census --algorithm=LogisticRegressionMitigation --sensitive_index=8 --max_iter=150
python main_mutation.py --dataset=credit --algorithm=LogisticRegressionMitigation --sensitive_index=9 --max_iter=2000
python main_mutation.py --dataset=bank --algorithm=LogisticRegressionMitigation --sensitive_index=1 --max_iter=100
python main_mutation.py --dataset=compas --algorithm=LogisticRegressionMitigation --sensitive_index=1 --max_iter=500
python main_mutation.py --dataset=compas --algorithm=LogisticRegressionMitigation --sensitive_index=3 --max_iter=500
```
```
python main_mutation.py --dataset=census --algorithm=TreeRegressorMitigation --sensitive_index=9 --max_iter=100
python main_mutation.py --dataset=census --algorithm=TreeRegressorMitigation --sensitive_index=8 --max_iter=100
python main_mutation.py --dataset=credit --algorithm=TreeRegressorMitigation --sensitive_index=9 --max_iter=200
python main_mutation.py --dataset=bank --algorithm=TreeRegressorMitigation --sensitive_index=1 --max_iter=50
python main_mutation.py --dataset=compas --algorithm=TreeRegressorMitigation --sensitive_index=1 --max_iter=400
python main_mutation.py --dataset=compas --algorithm=TreeRegressorMitigation --sensitive_index=3 --max_iter=400
```
```
python main_mutation.py --dataset=census --algorithm=SVM_Mitigation --sensitive_index=9 --max_iter=50
python main_mutation.py --dataset=census --algorithm=SVM_Mitigation --sensitive_index=8 --max_iter=50
python main_mutation.py --dataset=credit --algorithm=SVM_Mitigation --sensitive_index=9 --max_iter=1000
python main_mutation.py --dataset=bank --algorithm=SVM_Mitigation --sensitive_index=1 --max_iter=50
python main_mutation.py --dataset=compas --algorithm=SVM_Mitigation --sensitive_index=1 --max_iter=300
python main_mutation.py --dataset=compas --algorithm=SVM_Mitigation --sensitive_index=3 --max_iter=300
```
```
python main_mutation.py --dataset=census --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=9 --max_iter=100
python main_mutation.py --dataset=census --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=8 --max_iter=100
python main_mutation.py --dataset=credit --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=9 --max_iter=2500
python main_mutation.py --dataset=bank --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=1 --max_iter=100
python main_mutation.py --dataset=compas --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=1 --max_iter=100
python main_mutation.py --dataset=compas --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=3 --max_iter=100
```
Here we run SMBO optimization technique for logistic regression and decision tree:
```
python main_SMBO.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9
python main_SMBO.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9
```
```
python main_SMBO.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8
python main_SMBO.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=8
```
```
python main_SMBO.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=1
python main_SMBO.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=1
```
```
python main_SMBO.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=3
python main_SMBO.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=3
```
```
python main_SMBO.py --dataset=credit --algorithm=LogisticRegression --sensitive_index=9
python main_SMBO.py --dataset=credit --algorithm=Decision_Tree_Classifier --sensitive_index=9
```
```
python main_SMBO.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1
python main_SMBO.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1
```
