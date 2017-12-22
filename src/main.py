import tensorflow as tf

from network import Network
from config import Config
from parser import Parser
import resnet_model

#RNN controller
args = Parser().get_parser().parse_args()
config = Config(args)
net = Network(config)
outputs,prob = net.neural_search()
hyperparams = net.gen_hyperparams(outputs)

# Model

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(hyperparams))
