FROM ubuntu:18.04

MAINTAINER Saeid Tizpaz-Niari <saeid@utep.edu>

# Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update
RUN apt-get -y install git build-essential python3 python3-pip libjpeg-dev zlib1g-dev vim nano texlive-latex-base texlive-latex-recommended texlive-pictures screen
RUN pip3 install numpy scipy matplotlib pydotplus scikit-learn fairlearn 
# Setup PARFAIT-ML
WORKDIR /root
ADD adf_data /root/parfait-ml/adf_data
ADD adf_utils /root/parfait-ml/adf_utils
ADD datasets /root/parfait-ml/datasets
ADD README.md /root/parfait-ml/
ADD *.xml /root/parfait-ml/
ADD *.py /root/parfait-ml/
WORKDIR /root/parfait-ml/
RUN python3 main_random.py --dataset=census --algorithm=LogisticRegression --sensitive_index=9 --max_iter=100000  2>&1 | tee LogisticRegression_census_gender_random_output.txt &
WORKDIR /root/parfait-ml/