[![DOI](https://zenodo.org/badge/???)](???) 

# Parafait-ML: (PARameter FAIrness Testing for ML Libraries)
This repository provides the tool and the evaluation subjects for the paper "Fairness-aware Configuration of Machine Learning Libraries" accepted for the technical track at [ICSE'2022](https://conf.researchr.org/track/icse-2022/icse-2022-papers).
A pre-print of the paper is available [here](https://drive.google.com/file/d/1uafhRhKCBZLoEo8ledg343uQR2H24eHe).

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


## General Instructions: How to apply Parfait-ML on new subjects


## Developer and Maintainer
* **Saeid Tizpaz-Niari** (saeid at utep.edu)


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
