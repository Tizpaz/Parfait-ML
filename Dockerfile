FROM ubuntu:20.04

# Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update
RUN apt-get -y install git build-essential python3 python3-pip libjpeg-dev zlib1g-dev vim nano texlive-latex-base texlive-latex-recommended texlive-pictures screen graphviz
RUN pip3 install numpy scipy matplotlib pydotplus scikit-learn==0.24.0 fairlearn 
# Setup PARFAIT-ML
WORKDIR /root
ADD Dataset /root/parfait-ml/Dataset
ADD Results /root/parfait-ml/Results
ADD subjects /root/parfait-ml/subjects
ADD README.md /root/parfait-ml/
#ADD *.xml /root/parfait-ml/
ADD *.py /root/parfait-ml/
ADD *.sh /root/parfait-ml/
WORKDIR /root/parfait-ml/
RUN python3 main_mutation.py --dataset=census --algorithm=LogisticRegression --sensitive_index=8 --output=LogisticRegression_census_sex.csv --time_out=10
WORKDIR /root/parfait-ml/