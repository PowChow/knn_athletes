#!/usr/bin/env bash

cd ~
wget http://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh
bash Anaconda2-4.1.1-Linux-x86_64.sh -b
echo 'PATH="/home/ubuntu/anaconda2/bin:$PATH"' >> .bashrc
. .bashrc
source ~/.bashrc
