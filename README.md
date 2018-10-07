# Searching for activation functions 

This project attempts to implement NIPS 2017 paper "Searching for activation function" (Zoph & Le 2017). Although neural networks are powerful and flexible models, they are still hard to design and limited by human creativity. Using a combination of exhaustive and reinforcement learning-based search, the paper claims to be able to discover multiple novel activation functions. We tried to verify the claims of this paper by replicating the original study. However we were unable to get good results as probably due to the lack of massive computing resources used in the original experiment (800 Titan X GPUs).   

![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/nas.jpeg)

# Dependencies 

- Anaconda3
- TensorFlow-GPU >=1.4

# Setting up the docker environment
If you do not have the right dependencies to run this project, you can use our docker image which we used too to run these experiments on. 
```
docker pull etheleon/dotfiles
docker run --runtime=nvidia -it etheleon/dotfiles
```


# Running the code
Do a git clone of the repo first, then navigate into the src folder where the code of this project is stored
``` 
git clone https://github.com/Neoanarika/Searching-for-activation-functions.git
cd Searching-for-activation-functions
cd src
```
Download the data first, then find the activation functions
```
python cifar10_download_and_extract.py
python main.py
```

Next, test against your newly generated activation functions 
```
python cifar100_download_and_extract.py
python cifar100_train.py
python cifar100_test.py
```

Or you can open up the jupyter notebook in the repo and run from there. 

# RNN controller 

![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/Rnn.png)
![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/graph.png)

# Some sample activation functions found
| Activation functions  |
| ------------- |
| 3x  |
| 1  |
| -3  |

Clearly we are doing something wrong, the problem with implementing these papers is that even if it doesn't work, it could be due to us not running it long enough, or perhaps there's a bug in the program that we are unaware of that is causing the negative result. 

# Evaluating Swish
We also implemented swish, which was the activaiton function found and discussed in the original paper

![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/swish_.png)

```
python swish.py
```

![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/src/img/loss_rmsprop.png)

We found a few things, the first is that sometimes during the inital phase of training, the loss function remains the same on average. This shows that swish suffers from poor intialisation during training, at least when using initally normal distributed weights with std_dev =0.1. We tried various initialisations but there were no improvements found. Finially changing the optimiser from SGD to Rmsprop solved the problem. The diagram above is from training with Rmsprop. 


# Visualising Swish activation function
![alt text](https://github.com/Neoanarika/Searching-for-activation-functions/blob/master/img/swish_com.png)

Swish has a sharp global minima especially when compared with Relu, which may account for the high variance of the gradient updates as the model might be stuck in the wedge to reach the global minima. Learning rate decay might thus help improve the training for models using swish. Furthermore a sharper minima corresponds with poorer generalisation, which might explain why it performs slightly worse than relu in practise. 

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

