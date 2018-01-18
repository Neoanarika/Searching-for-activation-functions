# Searching for activation functions 

This is my tensorflow implementation of the Google brain paper "Searching for activation functions" for the 2017 NIPS challenge. 

# How to use 

Download the data first then find the activation functions
```
python cifar10_download_and_extract.py
python cifar10_main.py
```

# RNN controller 

![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/Rnn.png)
![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/graph.png)

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
Status : Still a work in progress.

# Dependencies 

- Python 3
- TensorFlow-GPU >=1.4

