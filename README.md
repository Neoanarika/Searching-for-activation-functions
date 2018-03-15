# Searching for activation functions 

This project attempts to implement NIPS 2017 paper "Searching for activation function" (Zoph & Le 2017). Although neural networks are powerful and flexible models they are still hard to design and limited to human creativity. Using a combination of exhaustive and reinforcement learning-based search, the paper claims to be able to discover multiple novel activation functions. We try to verify the claims of the paper by trying to replicate the original study. However we were unable to get a good results probably because of the lack of massive computing resources used in the original experiment (800 Titan X GPUs).   

![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/nas.jpeg)

# How to use 
First git clone the repo and then to use the code for this project is stored in src folder, so cd into the src folder 
``` 
git clone https://github.com/Neoanarika/Searching-for-activation-functions.git
cd Searching-for-activation-functions
cd src
```
Download the data first then find the activation functions
```
python cifar10_download_and_extract.py
python main.py
```

to test against ur newly generated activation functions 
```
python cifar100_download_and_extract.py
python cifar100_train.py
python cifar100_test.py
```

Or you can open up the jupyter notebook in the repo and run from there. 

# RNN controller 

![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/Rnn.png)
![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/graph.png)

# Swish
We also implemented swish which was the activaiton function found and discussed in the original paper

![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/swish_.png)

![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/swish_graph.png)

```
python swish.py
```

![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/src/img/loss_rmsprop.png)

# Visualising Swish activation function
![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/swish_com.png)

# Citation
```
@article{DBLP:journals/corr/abs-1710-05941,
  author    = {Prajit Ramachandran and
               Barret Zoph and
               Quoc V. Le},
  title     = {Searching for Activation Functions},
  journal   = {CoRR},
  volume    = {abs/1710.05941},
  year      = {2017},
  url       = {http://arxiv.org/abs/1710.05941},
  archivePrefix = {arXiv},
  eprint    = {1710.05941},
  timestamp = {Wed, 01 Nov 2017 19:05:42 +0100},
  biburl    = {http://dblp.org/rec/bib/journals/corr/abs-1710-05941},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

# Dependencies 

- Python 3
- TensorFlow-GPU >=1.4

