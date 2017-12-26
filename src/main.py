import os
import tensorflow as tf

from network import Network
from config import Config
from parser import Parser
from resnet_model import resnet
from dataset import DataSet

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3

#128 because batch_size
X = tf.placeholder(tf.float32, shape=[128,_HEIGHT, _WIDTH, _DEPTH])

#RNN controller
args = Parser().get_parser().parse_args()
config = Config(args)
net = Network(config)
outputs,prob = net.neural_search()
hyperparams = net.gen_hyperparams(outputs)

# Generate hyper params for the model
sess = tf.Session()
sess.run(tf.global_variables_initializer())
activations = sess.run(hyperparams)

# Set model
# 2 is the number of units
resnetwork = resnet(2,activations)
#Number of classes in cifar10
model = resnetwork.cifar10_resnet_v2_generator(20,10)
model(X,is_training = True)
data = DataSet(config)
for X, Y, tot in data.next_batch("train"):
    print(X)
