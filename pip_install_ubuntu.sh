# KEEP UBUNTU OR DEBIAN UP TO DATE
# Install components for python environment 
# Copy this script into nano or pip from local into instance
# cat ~/.ssh/id_rsa.pub | ssh -i ~/.ssh/class_key.pem ubuntu@ec2-xx-xx-xxx.us-west-2.compute.amazonaws.com 'cat >> wine.csv'

sudo apt-get install -y update
sudo apt-get install -y upgrade
sudo apt-get install -y dist-upgrade
sudo apt-get install -y autoremove
sudo apt-get install python-setuptools python-dev build-essential
sudo easy_install pip

sudo apt-get install git

git clone https://github.com/PowChow/knn_athletes.git
sudo pip install -r ~/knn_athletes/requirements.txt
