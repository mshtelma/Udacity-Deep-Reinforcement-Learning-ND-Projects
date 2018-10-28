#!/bin/sh
wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh
sh ./Anaconda3-5.3.0-Linux-x86_64.sh
source ~/.bashrc

sudo apt install unzip
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip
unzip Reacher_Linux_NoVis.zip


conda create -y -n deeprl python=3.6 anaconda
source activate deeprl
conda install -y pytorch torchvision -c pytorch
conda install -y opencv scikit-image
#conda uninstall -y ffmpeg # needed for gym monitor
#conda install -y -c conda-forge opencv ffmpeg  # needed for gym monitor
#pip install torchsummary tensorboardX dill gym Box2D box2d-py unityagents
pip install torchsummary tensorboardX unityagents
#cd ..

git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[atari]'
cd ..

git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents/ml-agents
pip install .
cd ..