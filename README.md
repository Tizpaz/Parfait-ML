# Parafait-ML: (PARameter FAIrness Testing for ML Libraries)
This repository provides the tool and the evaluation subjects for the paper "Fairness-aware Configuration of Machine Learning Libraries" accepted for the technical track at [ICSE'2022](https://conf.researchr.org/track/icse-2022/icse-2022-papers).

The repository includes:
* a [Dockerfile](Dockerfile) to build the Docker script,
* a set of required libraries for running the tool on local machine,
* the source code of *Parfait-ML*,
* the evaluation subjects: [subjects](./subjects),
* the pre-built evaluation all results: [Dataset](./Dataset), and
* the scripts to rerun all search experiments: [scripts](./script.sh).

## Docker Image
A pre-built version of *Parfait-ML* is also available as [Docker image](https://hub.docker.com/r/????):
```
docker pull ???
docker run -it --rm ???
```

We recommend to use Docker's [volume](https://docs.docker.com/engine/reference/run/#volume-shared-filesystems) feature to connect the docker container to the own file system so that Parfait-ML's results can be easily accessed.
Furthermore, we recommend to run scripts in an own [screen session](https://linuxize.com/post/how-to-use-linux-screen/#starting-named-session) so that the results can be observed during execution.

## Tool
*Parfait-ML* is a search-based software testing and statistical debugging tool to configure ML libraries fairly and detect fairness bugs in the configuration space of ML algorithms. *Parfait-ML* focuses on the hyperparameters in training of ML models and whether they amplify or suppress discriminations in data-driven software. Some prominent examples of hyperparameters include l1 vs. l2 loss
function in support vector machines, the maximum depth of a decision tree,
and the number of layers/neurons in deep neural networks.

*Parfait-ML* consists of two components: 1) search algorithms 2) data-driven explanations. The tool has three search algorithms to explore the space of ML hyperparameters in order to find what hyperparameter values can lead to high and low fairness. The current experiments include five ML algorithms and their hyperparameter spaces (e.g., logistic regression). To train ML algorithms, Parfait-ML includes four social-critical datasets (e.g., adult census income). During the search, the tool consider the default configuration of ML algorithm and its performance as a lower-bound on the functional accuracy requirements. Then, it selects inputs that satisfy the accuracy requirements and lead to an interesting fairness value. In addition, the coverage-based algorithm (graybox search) uses feedback from program internals to identify interesting inputs. For the fairness metric, we use EOD: the difference in true positive rates between two groups, and AOD: the average difference between true positive rates and false positive rates. We note that these metrics measure biases and higher values mean lower fairness. 

After stopping the search algorithm (a default of 4 hours), we use clustering
algorithms to partition the hyperparameter inputs generated from the search based on fairness and accuracy values. Given two groups of inputs with low and high biases and similar accuracy, we infer CART decision trees that show what hyperparameters are common inside a cluster and what hyperparameters distinguish low fairness configurations from the high fairness one. 

### Requirements
* Python 3.6
* pip, libjpeg-dev (Ubuntu) or Grphviz (MacOS)
* numpy, scipy, matplotlib, pydotplus, scikit-learn, fairlearn (installed with pip3)

### How to setup Parfait-ML
If you use the pre-built [Docker image](#docker-image), the tool is already built and ready to use so that you can skip this section. Otherwise, the installation of required packlages and libraries should be sufficient to run Parfait-ML.


### Getting Started with an example
After succesfully setup *Parfait-ML*, you can try a simple example to check the basic functionality.
We prepared a simple run script for the logistic regression subject over census dataset with race as sensitive attribute. The [`black-box fuzzer`](./main_mutation.py) interacts with the
[`logistic regression algorithm`](subjects/LogisticRegression.py). The hyperparameter variables and their domains are specified
as an XML file, see [`XML file for Logistic Regression`](subjects/LogisticRegression_Params.XML). The datasets are in [`datasets folder`](subjects/datasets/). For this example, we use [`census dataset`](subjects/datasets/census). 

We first generate test cases. Throughout the paper, we run
the test-case generation procedure for 4, except for RQ4.
Here, we run the black-box fuzzer for 10 minutes:
```
python3 main_mutation.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --output=LogisticRegression_census_sex.csv --time_out=600
```
where the column with index 8 is corresponds to `race` attribute of each sample individual. After $10$ minutes, the process
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
python clustering_DT.py --test_case=LogisticRegression_census_sex.csv --clusters 2
```
where `LogisticRegression_census_sex.csv` is the name of test-case outcome from the 
last step and `clusters` show the number of cluster. This
will produce clustering and decision tree models that can find
inside `Results` folders (two png files: one is for clustering
result and another is for decision tree).

### Complete Evaluation Reproduction
We include the script to run the search algorithms for the
entire dataset:
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
python clustering_DT.py --test_case=TreeRegressor_census_sex.csv --clusters 3
python clustering_DT.py --test_case=TreeRegressor_census_race.csv --clusters 2
```
Note: we include the experiment used in this paper for 10 runs in Dataset
folder. As an example for sex and race:
```
python clustering_DT.py --test_case=Run1/TreeRegressor_census_gender_mutation_res.csv --clusters 3
python clustering_DT.py --test_case=Run1/TreeRegressor_census_race_mutation_res.csv --clusters 2
```

#### Table 2 (RQ 1)
To produce the Table 2, please simply issue:
```
python3 RQ-1.py
```
The program goes through each experiment with 10 runs, perform
statistical analysis over the repeated experiemtns, and generate
RQ1 table. The file used in the paper can be found in [`RQ1`](Results/RQ-Dataset/RQ1.csv).

#### Table 3 and Figure 3 (RQ 2)
To produce the Table 3, please simply issue:
```
python3 RQ-2.py
```
The program will initiate and update the table RQ2 as it goes
through the experiments. At the end, it will also calculate 95%
temporal progress of search algorithms as shown in Figure 3.
`NOTE`: it might take over 1 hour
to complete the calculation. The table obtained in the paper can be found in [`RQ2`](Results/RQ-Dataset/RQ2.csv) and the temporal fuzzing can be found
in [`Fuzzing Progress`](Results/Fuzzing_Progress).

#### Figure 4 (RQ 3)
To produce Figure 3 and Figure 4, please simply issue:
```
python3 RQ-3.py
```
The program will create RQ1 to RQ10 folders, and each folder
includes clustering and decision tree models for each experiment.
`NOTE`: it might take over 1 hour
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
results used for Table 5. The results reported in the paper are
included [`RQ4-Exp`](Results/RQ-Dataset/RQ4-exp-6(m).csv) and
[`RQ4-SMBO`](Results/RQ-Dataset/RQ4-SMBO.csv).

## How to apply Parfait-ML on new subjects
Our framework allows a simple extension to include new dataset application as well as new algorithm. Currently, we support
5 dataset applications and 10 distinct ML algorithms. 

To add a new dataset, the user needs to import the dataset into
the [`subject/datasets`](subjects/datasets) and then provide an
abstract class in the [`abstract dataset classes`](subjects/adf_data) similar to existing 5 datasets. Then, the user
can build an instance of dataset in the search algorithms,
e.g., see line 23 in [`random search`](main_random.py).

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
