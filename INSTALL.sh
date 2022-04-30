#!/bin/bash
# Installation for Ubuntu (similar packages are required for MacOS)
apt-get -y update
apt-get -y install git build-essential python3 python3-pip libjpeg-dev zlib1g-dev vim nano texlive-latex-base texlive-latex-recommended texlive-pictures screen graphviz
pip3 install numpy scipy matplotlib pydotplus scikit-learn==0.24.0 fairlearn 
