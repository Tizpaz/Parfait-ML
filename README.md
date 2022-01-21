[![DOI](https://zenodo.org/badge/???)](???) 

# Parafait-ML: (PARameter FAIrness Testing for ML Libraries)
This repository provides the tool and the evaluation subjects for the paper "Fairness-aware Configuration of Machine Learning Libraries" accepted for the technical track at [ICSE'2022](https://conf.researchr.org/track/icse-2022/icse-2022-papers).

The repository includes:
* a [Dockerfile](Dockerfile) to build the Docker script,
* a set of required libraries for running the tool on local machine,
* the source code of *Parfait-ML*,
* the evaluation subjects: [subjects](./subjects),
* the pre-built evaluation all results: [Dataset](./Dataset), and
* the scripts to rerun all experiments: [scripts](./script.sh).

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
Therefore, we prepared a simple run script for the logistic regression subject over census dataset with race as sensitive attribute. The [`black-box fuzzer`](./main_mutation.py) interacts with the
[`logistic regression algorithm`](subjects/LogisticRegression.py). The hyperparameter variables and their domains are specified
as an XML file, see [`XML file for Logistic Regression`](subjects/LogisticRegression_Params.XML). The datasets are in [`datasets folder`](subjects/datasets/). For this example, we use [`census dataset`](subjects/datasets/census). 

We first generate test cases. Throughout the paper, we run
the test-case generation procedure for 4, except for RQ4.
Here, we run the black-box fuzzer for 10 minutes:
```
python3 main_mutation.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --time_out=600
```
where the column with index 8 is corresponds to `race` attribute of each sample individual. After $10$ minutes, the process
ends and shows the test case filename, which is
a CSV file. Let us name the file `csv_test_file`.
We can issue `vi` command to explore the csv
file (`Dataset/csv_test_file`) where each row is a configuration of hyperparameter
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
python clustering_DT.py --test_case=csv_test_file --clusters 2
```
where `csv_test_file` is the name of test-case outcome from the 
last step and `clusters` show the number of cluster. This
will produce clustering and decision tree models that can find
inside `Results` folders (two png files with `clustered` and `tree`
strings in the name).

### Complete Evaluation Reproduction
Explain scripts that can run and generate all results...

```
???
```

#### Table 1

#### ...


## General Instructions: How to apply Parfait-ML on new subjects



## License
This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details


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
[this link](https://drive.google.com/drive/folders/13figGs64BcPwwcLvQRKsQMz71wcU4UQw).
