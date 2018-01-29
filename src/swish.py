# Module 7: Convolutional Neural Network (CNN)
# CNN model with dropout for MNIST dataset

# CNN structure:
# · · · · · · · · · ·      input data                                               X  [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1x4  stride 1                            W1 [5, 5, 1, 4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                               Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4x8  with max pooling stride 2           W2 [5, 5, 4, 8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                                 Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8x12 stride 2 with max pooling stride 2  W3 [4, 4, 8, 12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                                                   Y3 [batch, 7, 7, 12]
#      \x/x\x\x/        -- fully connected layer (relu)                             W4 [7*7*12, 200]
#       · · · ·                                                                     Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)                          W5 [200, 10]
#        · · ·                                                                      Y [batch, 10]

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Hyper Parameters
learning_rate = 0.01
training_epochs = 2
batch_size = 100

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True,reshape=False,validation_size=0)
logdir = '/users/mingliangang/Desktop/grad_cnn'
# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
pkeep = tf.placeholder(tf.float32)

L1 = 4 # first convolutional filters
L2 = 8 # second convolutional filters
L3 = 12 # third convolutional filters
L4 = 200 # fully connected neurons

W1 = tf.Variable(tf.truncated_normal([5,5,1,L1], stddev=0.08))
B1 = tf.Variable(tf.zeros([L1]))
beta1 =  tf.Variable(tf.truncated_normal([1], stddev=0.08))
W2 = tf.Variable(tf.truncated_normal([5,5,L1,L2], stddev=0.08))
B2 = tf.Variable(tf.zeros([L2]))
beta2 =  tf.Variable(tf.truncated_normal([1], stddev=0.08))
W3 = tf.Variable(tf.truncated_normal([4,4,L2,L3], stddev=0.08))
B3 = tf.Variable(tf.zeros([L3]))
beta3 =  tf.Variable(tf.truncated_normal([1], stddev=0.08))
W4 = tf.Variable(tf.truncated_normal([7*7*L3,L4], stddev=0.08))
B4 = tf.Variable(tf.zeros([L4]))
W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.08))
B5 = tf.Variable(tf.zeros([10]))
#tf.summary.scalar('W1',tf.reduce_mean(W1))

# Step 2: Setup Model
x1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') + B1
Y1 = x1*tf.nn.sigmoid(beta1*x1)# output is 28x28
x2 = tf.nn.conv2d(Y1, W2, strides=[1,1,1,1], padding='SAME') + B2
Y2 = x2*tf.nn.sigmoid(beta2*x2)
Y2 = tf.nn.max_pool(Y2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # output is 14x14
Y2= tf.nn.dropout(Y2, pkeep)
x3 = tf.nn.conv2d(Y2, W3, strides=[1,1,1,1], padding='SAME') + B3
Y3 = x3*tf.nn.sigmoid(beta3*x3)
Y3 = tf.nn.max_pool(Y3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # output is 7x7
Y3= tf.nn.dropout(Y3, pkeep)

# Flatten the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * L3])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
#YY4 = tf.nn.dropout(Y4, 0.3)
Ylogits = tf.matmul(Y4, W5) + B5
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y))
tf.summary.scalar('loss',loss)
# Step 4: Optimizer
#optimizer = tf.train.RMSPropOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)
#optimizer = tf.train.AdamOptimizer()
grad = optimizer.compute_gradients(loss)
tf.summary.scalar('beta1',tf.reduce_mean(beta1))
tf.summary.scalar('grad',tf.reduce_mean(grad[0][0]))
tf.summary.scalar('W1',tf.reduce_mean(grad[0][1]))
tf.summary.histogram('grad',tf.reduce_mean(grad[0][0]))
tf.summary.histogram('W1',tf.reduce_mean(grad[0][1]))
tf.summary.histogram('beta1',tf.reduce_mean(beta1))

train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir+ '/train',
                                      sess.graph)
sess.run(init)

# Step 5: Training Loop
for epoch in range(training_epochs):
    num_batches = int(mnist.train.num_examples / batch_size)
    for i in range(num_batches):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        train_data = {X: batch_X, y: batch_y, pkeep: 0.5}
        summary,_ = sess.run([merged,train], feed_dict=train_data)
        writer.add_summary(summary,epoch*num_batches+i+1)
        print(epoch * num_batches + i + 1, "Training accuracy =", sess.run(accuracy, feed_dict=train_data),
          "Loss =", sess.run(loss, feed_dict=train_data))

# Step 6: Evaluation
test_data = {X:mnist.test.images,y:mnist.test.labels, pkeep: 1.0}
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))
