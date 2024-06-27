# FairLay-ML

This repository provides the tool and the evaluation subjects for the UG thesis: [FairLay-ML: Intuitive Remedies for Unfairness in Data-Driven Social-Critical Algorithms](https://arxiv.org/abs/2307.05029). FairLay-ML is built on top of "Fairness-aware Configuration of Machine Learning Libraries" accepted for the technical track at [ICSE'2022](https://arxiv.org/abs/2202.06196).

The repository includes:
* a [Dockerfile](Dockerfile) to build the Docker script,
* a set of required libraries for running the tool on the local machine,
* the source code of *Parfait-ML*,
* the evaluation subjects: [subjects](./subjects),
* the pre-built evaluation all results: [Dataset](./Dataset), and
* the scripts to rerun all search experiments: [scripts](./script.sh).

## Setup
FairLay-ML uses Google Cloud Platforms C2 (compute optimized, 8 vCPU, 32 GB) Computer Engine Virtual Machine.
The installation process is as follows:

1) Follow this tutorial for graphics displays: https://ubuntu.com/blog/launch-ubuntu-desktop-on-google-cloud

2) Follow this tutorial for Docker container installation: https://docs.docker.com/engine/install/ubuntu/

3) sudo apt-get install git

4) Install required packages for python

5) Install jupyter notebook with 'sudo apt-get install jupyter' and set up jupyter notebook following this tutorial: https://tudip.com/blog-post/run-jupyter-notebook-on-google-cloud-platform/ (try to set up security as well using a password)

Here are some other considerations:
* Streamlit "server" can be run from "monitoring_framework"
* Found some information about the census dataset here: https://archive.ics.uci.edu/ml/datasets/census+income
* Found some information about the German credit here: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
* Found some information about the bank here: https://archive.ics.uci.edu/ml/datasets/bank+marketing

## Docker File
```
docker run --rm -it  parfait:1.0.0
```
We recommend using Docker's [volume](https://docs.docker.com/engine/reference/run/#volume-shared-filesystems) feature to connect the docker container to the own file system so that Parfait-ML's results can be easily accessed.
Furthermore, we recommend running scripts in their own [screen session](https://linuxize.com/post/how-to-use-linux-screen/#starting-named-session) so that the results can be observed during execution.

Note: The tool is initially built for MacOS. Docker version
with Ubuntu might experience some unexpected errors.

## Parfait-ML
Users can modify *Parfait-ML* which is a search-based software testing and statistical debugging tool to configure ML libraries fairly and detect fairness bugs in the configuration space of ML algorithms. *Parfait-ML* focuses on the hyperparameters in the training of ML models and whether they amplify or suppress discriminations in data-driven software. Some prominent examples of hyperparameters include l1 vs. l2 loss
function in support vector machines, the maximum depth of a decision tree,
and the number of layers/neurons in deep neural networks.

*Parfait-ML* consists of two components: 1) search algorithms 2) data-driven explanations. The tool has three search algorithms to explore the space of ML hyperparameters in order to find what hyperparameter values can lead to high and low fairness. The current experiments include five ML algorithms and their hyperparameter spaces (e.g., logistic regression). To train ML algorithms, Parfait-ML includes four social-critical datasets (e.g., adult census income). During the search, the tool consider the default configuration of ML algorithm and its performance as a lower-bound on the functional accuracy requirements. Then, it selects inputs that satisfy the accuracy requirements and lead to an interesting fairness value. In addition, the coverage-based algorithm (graybox search) uses feedback from program internals to identify interesting inputs. For the fairness metric, we use EOD: the difference in true positive rates between two groups, and AOD: the average difference between true positive rates and false positive rates. We note that these metrics measure biases and higher values mean lower fairness. 

After stopping the search algorithm (a default of 4 hours), we use clustering
algorithms to partition the hyperparameter inputs generated from the search based on fairness and accuracy values. Given two groups of inputs with low and high biases and similar accuracy, we infer CART decision trees that show what hyperparameters are common inside a cluster and what hyperparameters distinguish low fairness configurations from the high fairness one. 

### Requirements
* Python 3.7
* pip, libjpeg-dev (Ubuntu) or Grphviz (MacOS)
* numpy, scipy, matplotlib, pydotplus, scikit-learn==0.24.0, fairlearn (installed with pip3)

### How to setup Parfait-ML
If you use the pre-built [Docker image](#docker-image), the tool is already built to use in Ubuntu base. Otherwise, the installation of required packlages and libraries should be sufficient to run Parfait-ML. Note: the tool is substantially tested in MacOS system.


### Getting Started with an example
After successfully setup *Parfait-ML*, you can try a simple example to check the basic functionality.
We prepared a simple run script for the logistic regression subject over census dataset with race as sensitive attribute. The [`black-box fuzzer`](./main_mutation.py) interacts with the
[`logistic regression algorithm`](subjects/LogisticRegression.py). The hyperparameter variables and their domains are specified
as an XML file, see [`XML file for Logistic Regression`](subjects/LogisticRegression_Params.XML). The datasets are in [`datasets folder`](subjects/datasets/). For this example, we use [`census dataset`](subjects/datasets/census). 

We first generate test cases. Throughout the paper, we run
the test-case generation procedure for 4, except for RQ4.
Here, we run the black-box fuzzer for 10 minutes:
```
python3 main_mutation.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --output=LogisticRegression_census_sex.csv --time_out=600
```
where the column with index 8 corresponds to `race` attribute of each sample individual. After $10$ minutes, the process
ends and shows the test case filename, which is
a CSV file. 
We can issue `vi` command to explore the csv
file (`Dataset/LogisticRegression_census_sex.csv`) where each row is a configuration of hyperparameter
values, their corresponding accuracy (score), AOD fairness metric,
EOD metric (TPR), etc. 

One can replace `main_mutation.py` with
`main_random.py` to run independently random algorithm
and with `main_coverage.py` to run graybox fuzzing algorithm.
Similarly, dataset and algorithms can be replaced to
perform experiments on other learning algorithms as well
as datasets (full lists are available in the paper
as well as in [`all scripts`](scripts.sh). Next,
we run clustering and decision tree algorithm:
```
python3 clustering_DT.py --test_case=LogisticRegression_census_sex.csv --clusters 2
```
where `LogisticRegression_census_sex.csv` is the name of test-case outcome from the 
last step and `clusters` show the number of cluster. This
will produce clustering and decision tree models that can find
inside `Results` folders (two png files: one is for clustering
result and another is for decision tree).

### Complete Evaluation Reproduction
We include the script to run the search algorithms for the
entire dataset (warning: it will run all experiments):
```
sh script.sh
```
By default, every experiment will run for 4 hours, and one needs
to repeat the experiments 10 times to reproduce RQ1 to RQ4.
The experiments used in the paper is included in the Dataset folder.

#### Figure 2 (mutation search + census with sex and race)
We need to first generate test cases using black-box mutation algorithm
for gender (sex) attribute wit index 9 (left) and race attribute with index 8 (right):
```
python3 main_mutation.py --dataset=census --algorithm=TreeRegressor --sensitive_index=9 --output=TreeRegressor_census_sex.csv --time_out=14400
python3 main_mutation.py --dataset=census --algorithm=TreeRegressor --sensitive_index=8 --output=TreeRegressor_census_race.csv --time_out=14400
```
Then, we need to apply clustering and decision tree inferences:
```
python3 clustering_DT.py --test_case=TreeRegressor_census_sex.csv --clusters 3
python3 clustering_DT.py --test_case=TreeRegressor_census_race.csv --clusters 2
```
These will generate clustering and decision tree png file. Note: we include the experiment used in this paper for 10 runs in Dataset folder. As an example for sex and race:
```
python3 clustering_DT.py --test_case=Run1/TreeRegressor_census_gender_mutation_res.csv --clusters 3
python3 clustering_DT.py --test_case=Run1/TreeRegressor_census_race_mutation_res.csv --clusters 2
```
The corresponding files can be found in ['Run1_DT'](Results/1.Decision_Tree_Run_1).

#### Table 2 (RQ 1)
To produce the Table 2, please simply issue:
```
python3 RQ-1.py
```
The program goes through each experiment with 10 runs, perform
statistical analysis over the repeated experiemtns, and generate
RQ1 table. This should take about one minute. The file used in the paper can be found in [`RQ1`](Results/RQ-Dataset/RQ1.csv).
```
vi Results/RQ-Dataset/RQ1.csv
```

#### Table 3 and Figure 3 (RQ 2)
To produce the Table 3, please simply issue:
```
python3 RQ-2.py
```
The program will initiate RQ-2 table,
update the table RQ2 as it goes
through the experiments, and calculate 95%
temporal progress of search algorithms as shown in Figure 3.
`NOTE`: it might take over 90 minutes
to complete the calculation. The table obtained in the paper can be found in [`RQ2`](Results/RQ-Dataset/RQ2.csv) and the temporal fuzzing can be found
in [`Fuzzing Progress`](Results/Fuzzing_Progress).
```
vi Results/RQ-Dataset/RQ2.csv 
```

#### Figure 4 (RQ 3)
To produce Figure 3 and Figure 4, please simply issue:
```
python3 RQ-3.py
```
The program will create RQ1 to RQ10 folders, and each folder
includes clustering and decision tree models for each experiment.
`NOTE`: it might take over 90 minutes
to complete the calculation.
The results used in the paper can be found from [`1.Decision_Tree_Run1`](Results/1.Decision_Tree_Run_1) to [`10.Decision_Tree_Run1`](Results/10.Decision_Tree_Run_10).
Also, the program will generate RQ3 table in [`RQ3`](Results/RQ-Dataset/RQ3.csv).
The table results have used to report statistics in section 6.6.

#### Table 4 and Table 5 (RQ 4)
To produce Table 4 and Table 5, please simply issue:
```
python3 RQ-4.py
```
The program will generate two csv files: `RQ4-exp-6(m).csv` file
shows the results used for Table 4 and `RQ4-SMBO.csv` file shows the
results used for Table 5.
```
vi Results/RQ4-exp-6(m).csv
vi Results/RQ4-SMBO.csv
```
The results reported in the paper are
included [`RQ4-Exp`](Results/RQ-Dataset/RQ4-exp-6(m).csv) and
[`RQ4-SMBO`](Results/RQ-Dataset/RQ4-SMBO.csv):


## How to apply Parfait-ML on new subjects
Our framework allows a simple extension to include new dataset application as well as new algorithm. Currently, we support
5 dataset applications and 10 distinct ML algorithms. 

To add a new dataset, the user needs to import the dataset into
the [`subject/datasets`](subjects/datasets) and then provide an
abstract class in the [`abstract dataset classes`](subjects/adf_data) similar to existing 5 datasets. Then, the user
can build an instance of dataset in the search algorithms
and add the dataset name to the dataset dictionary,
e.g., see lines 24, 318, and 319 in [`random search`](main_random.py).

To add a new algorithm, the user needs to add the
target program into the [`subject`](subjects/) folder, and
define the interface function, e.g., see line 7 for
[`Decision Tree`](subjects/Decision_Tree_Classifier.py).
The interface function includes the input file (will be fed
by the search algorithm, train data, test data, train label,
and sensitive feature). The next step is to introduce the
interface in the search drivers, e.g., see lines 100
to 150 in [`random search`](main_random.py) where the
driver connects to 10 existing algorithm.


## License
This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details
