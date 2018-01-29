# Searching for activation functions 

This project attempts to implement NIPS 2017 paper "Searching for activation function" (Zoph & Le 2017). Although neural networks are powerful and flexible models they are still hard to design and limited to human creativity. Partly inspired by AutoML Alpha that was realsed by google on 17 Jan 2018, this project aims to provide some starter code for deep learning researchers, students and hobbyist to get started with making their own DIY "AutoML" software and start exploring the idea of meta-learning or learning2learning in deep learning and AI. 


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

