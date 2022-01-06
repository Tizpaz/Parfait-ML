[![DOI](https://zenodo.org/badge/???)](???) 

# Parafait-ML: (PARameter FAIrness Testing for ML Libraries)
This repository provides the tool and the evaluation subjects for the paper "Fairness-aware Configuration of Machine Learning Libraries" accepted for the technical track at [ICSE'2022](https://conf.researchr.org/track/icse-2022/icse-2022-papers).
A pre-print of the paper is available [here](drive.google.com/file/d/1uafhRhKCBZLoEo8ledg343uQR2H24eHe).

The repository includes:
* a [Dockerfile](Dockerfile) to build the Docker script,
* a [setup](setup.sh) script for manual building of the tool,
* the source code of *Parfait-ML*,
* the evaluation subjects: [evaluation/subjects](./???),
* the summarized evaluation results: [evaluation/results](./???), and
* the scripts to rerun all experiments: [evaluation/scripts](./???).

## Docker Image
A pre-built version of *Parfait-ML* is also available as [Docker image](https://hub.docker.com/r/????):
```
docker pull ???
docker run -it --rm ???
```

We recommend to use Docker's [volume](https://docs.docker.com/engine/reference/run/#volume-shared-filesystems) feature to connect the docker container to the own file system so that Parfait-ML's results can be easily accessed.
Furthermore, we recommend to run scripts in an own [screen session](https://linuxize.com/post/how-to-use-linux-screen/#starting-named-session) so that the results can be observed during execution.

## Tool
*Parfait-ML* is a search-based software testing and statistical debugging tool to configure ML libraries fairly and detect fairness bugs in the configuration space of ML algorithms. More explanation of tool later (???).

### Requirements
* Python 3.6
* pip, libjpeg-dev (Ubuntu) or Grphviz (MacOS)
* numpy, scipy, matplotlib, pydotplus, scikit-learn, fairlearn (installed with pip3)

### How to setup Parfait-ML
If you use the pre-built [Docker image](#docker-image), the tool is already built and ready to use so that you can skip this section. Otherwise, the installation of required packlages and libraries should be sufficient to run Parfait-ML.


### Getting Started with an example
After succesfully setup *Parfait-ML*, you can try a simple example to check
the basic functionality.
Therefore, we prepared a simple run script for the *???* subject.
It represents ???.
You can find the run script here: [`evaluation/scripts/run_example.sh`](evaluation/scripts/run_example.sh).
We have constructed all run scripts in the way that the compartment in the beginning defines the run configurations:

### Complete Evaluation Reproduction
Explain scripts that can run and generate all results...

```
???
```

#### Table 1

#### ...


## General Instructions: How to apply QFuzz on new subjects


## Developer and Maintainer
* **Saeid Tizpaz-Niari** (saeid at utep.edu)


Run the random search for adult census dataset with gender as sensitive feature:
```
python3 main_random.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000  2>&1 | tee LogisticRegression_census_gender_random_output.txt &
python3 main_random.py --dataset=census --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000  2>&1 | tee TreeRegressor_census_gender_random_output.txt &
python3 main_random.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_census_gender_random_output.txt &
python3 main_random.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_gender_random_output.txt &
python3 main_random.py --dataset=census --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_census_gender_random_output.txt &
```
This will result into two files: .csv file shows inputs and outcome of training; .txt file writes extra details (not essential part of the analysis).    

Run the other commands with other datasets as the following:   
```
python3 main_mutation.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000 2>&1 | tee LogisticRegression_census_gender_mutation_output.txt &
python3 main_mutation.py --dataset=census --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000 2>&1 | tee TreeRegressor_census_gender_mutation_output.txt &
python3 main_mutation.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_census_gender_mutation_output.txt &
python3 main_mutation.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_gender_mutation_output.txt &
python3 main_mutation.py --dataset=census --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_census_gender_mutation_output.txt &
```
```
python3 main_coverage.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000  2>&1 | tee LogisticRegression_census_gender_coverage_output.txt &
python3 main_coverage.py --dataset=census --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000 2>&1 | tee TreeRegressor_census_gender_coverage_output.txt &
python3 main_coverage.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_census_gender_coverage_output.txt &
python3 main_coverage.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_gender_coverage_output.txt &
python3 main_coverage.py --dataset=census --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_census_gender_coverage_output.txt &
```
```
python3 main_random.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --max_iter=100000 2>&1 | tee LogisticRegression_census_race_random_output.txt &
python3 main_random.py --dataset=census --algorithm=TreeRegressor --sensitive_index=8 --max_iter=100000  2>&1 | tee TreeRegressor_census_race_random_output.txt &
python3 main_random.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=8 --max_iter=100000  2>&1 | tee Decision_Tree_Classifier_census_race_random_output.txt &
python3 main_random.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=8 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_race_random_output.txt &
python3 main_random.py --dataset=census --algorithm=SVM --sensitive_index=8 --max_iter=100000 2>&1 | tee SVM_census_race_random_output.txt &
```
```
python3 main_mutation.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --max_iter=100000 2>&1 | tee LogisticRegression_census_race_mutation_output.txt &
python3 main_mutation.py --dataset=census --algorithm=TreeRegressor --sensitive_index=8 --max_iter=100000 2>&1 | tee TreeRegressor_census_race_mutation_output.txt &
python3 main_mutation.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=8 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_census_race_mutation_output.txt &
python3 main_mutation.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=8 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_race_mutation_output.txt &
python3 main_mutation.py --dataset=census --algorithm=SVM --sensitive_index=8 --max_iter=100000 2>&1 | tee SVM_census_race_mutation_output.txt &
```
```
python3 main_coverage.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --max_iter=100000  2>&1 | tee LogisticRegression_census_race_coverage_output.txt &
python3 main_coverage.py --dataset=census --algorithm=TreeRegressor --sensitive_index=8 --max_iter=100000 2>&1 | tee TreeRegressor_census_race_coverage_output.txt &
python3 main_coverage.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=8 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_census_race_coverage_output.txt &
python3 main_coverage.py --dataset=census --algorithm=Discriminant_Analysis --sensitive_index=8 --max_iter=100000 2>&1 | tee Discriminant_Analysis_census_race_coverage_output.txt &
python3 main_coverage.py --dataset=census --algorithm=SVM --sensitive_index=8 --max_iter=100000 2>&1 | tee SVM_census_race_coverage_output.txt &
```
```
python3 main_random.py --dataset=credit --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000 2>&1 | tee LogisticRegression_credit_gender_random_output.txt &
python3 main_random.py --dataset=credit --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000 2>&1 | tee TreeRegressor_credit_gender_random_output.txt &
python3 main_random.py --dataset=credit --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_credit_random_random_output.txt &
python3 main_random.py --dataset=credit --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_credit_gender_random_output.txt &
python3 main_random.py --dataset=credit --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_credit_gender_random_output.txt &
```
```
python3 main_mutation.py --dataset=credit --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000 2>&1 | tee LogisticRegression_credit_gender_mutation_output.txt &
python3 main_mutation.py --dataset=credit --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000 2>&1 | tee TreeRegressor_credit_gender_mutation_output.txt &
python3 main_mutation.py --dataset=credit --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_credit_gender_mutation_output.txt &
python3 main_mutation.py --dataset=credit --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_credit_gender_mutation_output.txt &
python3 main_mutation.py --dataset=credit --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_credit_gender_mutation_output.txt &
```
```
python3 main_coverage.py --dataset=credit --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000 2>&1 | tee LogisticRegression_credit_gender_coverage_output.txt &
python3 main_coverage.py --dataset=credit --algorithm=TreeRegressor --sensitive_index=9 --max_iter=100000 2>&1 | tee TreeRegressor_credit_gender_coverage_output.txt &
python3 main_coverage.py --dataset=credit --algorithm=Decision_Tree_Classifier --sensitive_index=9 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_credit_gender_coverage_output.txt &
python3 main_coverage.py --dataset=credit --algorithm=Discriminant_Analysis --sensitive_index=9 --max_iter=100000 2>&1 | tee Discriminant_Analysis_credit_gender_coverage_output.txt &
python3 main_coverage.py --dataset=credit --algorithm=SVM --sensitive_index=9 --max_iter=100000 2>&1 | tee SVM_credit_gender_coverage_output.txt &
```
```
python3 main_random.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_bank_random_output.txt &
python3 main_random.py --dataset=bank --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_bank_random_output.txt &
python3 main_random.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_bank_random_output.txt &
python3 main_random.py --dataset=bank --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_bank_random_output.txt &
python3 main_random.py --dataset=bank --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_bank_random_output.txt &
```
```
python3 main_mutation.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_bank_mutation_output.txt &
python3 main_mutation.py --dataset=bank --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_bank_mutation_output.txt &
python3 main_mutation.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_bank_mutation_output.txt &
python3 main_mutation.py --dataset=bank --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_bank_mutation_output.txt &
python3 main_mutation.py --dataset=bank --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_bank_mutation_output.txt &
```
```
python3 main_coverage.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_bank_coverage_output.txt &
python3 main_coverage.py --dataset=bank --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_bank_coverage_output.txt &
python3 main_coverage.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_bank_coverage_output.txt &
python3 main_coverage.py --dataset=bank --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_bank_coverage_output.txt &
python3 main_coverage.py --dataset=bank --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_bank_coverage_output.txt &
```
```
python3 main_random.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_compas_gender_random_output.txt &
python3 main_random.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_compas_gender_random_output.txt &
python3 main_random.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_gender_random_output.txt &
python3 main_random.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_gender_random_output.txt &
python3 main_random.py --dataset=compas --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_compas_gender_random_output.txt &
```
```
python3 main_mutation.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_compas_gender_mutation_output.txt &
python3 main_mutation.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_compas_gender_mutation_output.txt &
python3 main_mutation.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_gender_mutation_output.txt &
python3 main_mutation.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_gender_mutation_output.txt &
python3 main_mutation.py --dataset=compas --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_compas_gender_mutation_output.txt &
```
```
python3 main_coverage.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=1 --max_iter=100000 2>&1 | tee LogisticRegression_compas_gender_coverage_output.txt &
python3 main_coverage.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=1 --max_iter=100000 2>&1 | tee TreeRegressor_compas_gender_coverage_output.txt &
python3 main_coverage.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=1 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_gender_coverage_output.txt &
python3 main_coverage.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=1 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_gender_coverage_output.txt &
python3 main_coverage.py --dataset=compas --algorithm=SVM --sensitive_index=1 --max_iter=100000 2>&1 | tee SVM_compas_gender_coverage_output.txt &
```
```
python3 main_random.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=3 --max_iter=100000  2>&1 | tee LogisticRegression_compas_race_random_output.txt &
python3 main_random.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=3 --max_iter=100000 2>&1 | tee TreeRegressor_compas_race_random_output.txt &
python3 main_random.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=3 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_race_random_output.txt &
python3 main_random.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=3 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_race_random_output.txt &
python3 main_random.py --dataset=compas --algorithm=SVM --sensitive_index=3 --max_iter=100000 2>&1 | tee SVM_compas_race_random_output.txt &
```
```
python3 main_mutation.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=3 --max_iter=100000 2>&1 | tee LogisticRegression_compas_race_mutation_output.txt &
python3 main_mutation.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=3 --max_iter=100000 2>&1 | tee TreeRegressor_compas_race_mutation_output.txt &
python3 main_mutation.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=3 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_race_mutation_output.txt &
python3 main_mutation.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=3 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_race_mutation_output.txt &
python3 main_mutation.py --dataset=compas --algorithm=SVM --sensitive_index=3 --max_iter=100000 2>&1 | tee SVM_compas_race_mutation_output.txt &
```
```
python3 main_coverage.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=3 --max_iter=100000 2>&1 | tee LogisticRegression_compas_race_coverage_output.txt &
python3 main_coverage.py --dataset=compas --algorithm=TreeRegressor --sensitive_index=3 --max_iter=100000 2>&1 | tee TreeRegressor_compas_race_coverage_output.txt &
python3 main_coverage.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=3 --max_iter=100000 2>&1 | tee Decision_Tree_Classifier_compas_race_coverage_output.txt &
python3 main_coverage.py --dataset=compas --algorithm=Discriminant_Analysis --sensitive_index=3 --max_iter=100000 2>&1 | tee Discriminant_Analysis_compas_race_coverage_output.txt &
python3 main_coverage.py --dataset=compas --algorithm=SVM --sensitive_index=3 --max_iter=100000 2>&1 | tee SVM_compas_race_coverage_output.txt &
```
Here, we combine our tool with ExponentiatedGradient approach from fairlearn library:
```
python3 main_mutation.py --dataset=census --algorithm=LogisticRegressionMitigation --sensitive_index=9 --max_iter=150
python3 main_mutation.py --dataset=census --algorithm=LogisticRegressionMitigation --sensitive_index=8 --max_iter=150
python3 main_mutation.py --dataset=credit --algorithm=LogisticRegressionMitigation --sensitive_index=9 --max_iter=2000
python3 main_mutation.py --dataset=bank --algorithm=LogisticRegressionMitigation --sensitive_index=1 --max_iter=100
python3 main_mutation.py --dataset=compas --algorithm=LogisticRegressionMitigation --sensitive_index=1 --max_iter=500
python3 main_mutation.py --dataset=compas --algorithm=LogisticRegressionMitigation --sensitive_index=3 --max_iter=500
```
```
python3 main_mutation.py --dataset=census --algorithm=TreeRegressorMitigation --sensitive_index=9 --max_iter=100
python3 main_mutation.py --dataset=census --algorithm=TreeRegressorMitigation --sensitive_index=8 --max_iter=100
python3 main_mutation.py --dataset=credit --algorithm=TreeRegressorMitigation --sensitive_index=9 --max_iter=200
python3 main_mutation.py --dataset=bank --algorithm=TreeRegressorMitigation --sensitive_index=1 --max_iter=50
python3 main_mutation.py --dataset=compas --algorithm=TreeRegressorMitigation --sensitive_index=1 --max_iter=400
python3 main_mutation.py --dataset=compas --algorithm=TreeRegressorMitigation --sensitive_index=3 --max_iter=400
```
```
python3 main_mutation.py --dataset=census --algorithm=SVM_Mitigation --sensitive_index=9 --max_iter=50
python3 main_mutation.py --dataset=census --algorithm=SVM_Mitigation --sensitive_index=8 --max_iter=50
python3 main_mutation.py --dataset=credit --algorithm=SVM_Mitigation --sensitive_index=9 --max_iter=1000
python3 main_mutation.py --dataset=bank --algorithm=SVM_Mitigation --sensitive_index=1 --max_iter=50
python3 main_mutation.py --dataset=compas --algorithm=SVM_Mitigation --sensitive_index=1 --max_iter=300
python3 main_mutation.py --dataset=compas --algorithm=SVM_Mitigation --sensitive_index=3 --max_iter=300
```
```
python3 main_mutation.py --dataset=census --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=9 --max_iter=100
python3 main_mutation.py --dataset=census --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=8 --max_iter=100
python3 main_mutation.py --dataset=credit --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=9 --max_iter=2500
python3 main_mutation.py --dataset=bank --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=1 --max_iter=100
python3 main_mutation.py --dataset=compas --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=1 --max_iter=100
python3 main_mutation.py --dataset=compas --algorithm=Decision_Tree_Classifier_Mitigation --sensitive_index=3 --max_iter=100
```
Here we run SMBO optimization technique for logistic regression and decision tree:
```
python3 main_SMBO.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9
python3 main_SMBO.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=9
```
```
python3 main_SMBO.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8
python3 main_SMBO.py --dataset=census --algorithm=Decision_Tree_Classifier --sensitive_index=8
```
```
python3 main_SMBO.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=1
python3 main_SMBO.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=1
```
```
python3 main_SMBO.py --dataset=compas --algorithm=LogisticRegression --sensitive_index=3
python3 main_SMBO.py --dataset=compas --algorithm=Decision_Tree_Classifier --sensitive_index=3
```
```
python3 main_SMBO.py --dataset=credit --algorithm=LogisticRegression --sensitive_index=9
python3 main_SMBO.py --dataset=credit --algorithm=Decision_Tree_Classifier --sensitive_index=9
```
```
python3 main_SMBO.py --dataset=bank --algorithm=LogisticRegression --sensitive_index=1
python3 main_SMBO.py --dataset=bank --algorithm=Decision_Tree_Classifier --sensitive_index=1
```
To reproduce the results for research questions, run the following commands.   
However, make sure that data from the runs are stored inside ``Dataset`` folder.   
The results will be saved in the current folder as ``.csv`` files or inside ``Results`` folder.    
To download the input data to run the following commands, use [this link](https://drive.google.com/drive/folders/1CVe5-tow5NiKynRDF1BFwU-gVJr1obh5).
```
python3 RQ-1.py
```
```
python3 RQ-2.py
```
```
python3 RQ-3.py
```
```
python3 RQ-4.py
```
The current results of these experiments are available at
[this link](https://drive.google.com/drive/folders/13figGs64BcPwwcLvQRKsQMz71wcU4UQw?usp=sharing).
