import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

#tensorflow make the library of MNIST because many people use thise standartic MNIST data..
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
#input_data.read_data_sets : at first execute this function download data , next times load data from MMIST_data/ directory

nb_classes = 10

X=tf.placeholder(tf.float32,[None,784]) #X matrix is 28*28 so 784 value exist in one x dataset
Y=tf.placeholder(tf.float32,[None,nb_classes]) #Y should be one-hot shape and classes will be 0 to 9

W=tf.Variable(tf.random_normal([784,nb_classes])) #X matrix has 784 data and 784 weights need to be revised
b=tf.Variable(tf.random_normal([nb_classes])) #one-hot shape output will be printed so each elements should be added by bias

hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)

cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct=tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs=30
batch_size=100

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_cost=0
		total_batch=int(mnist.train.num_examples/batch_size)

		for i in range(total_batch):
			batch_xs,batch_ys=mnist.train.next_batch(batch_size)
			c,_=sess.run([cost,optimizer],feed_dict={X:batch_xs,Y:batch_ys})
			avg_cost+=c/total_batch

		print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_cost))
 
	r=random.randint(0,mnist.test.num_examples-1)
	print("Label:",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
	print("Prediction:",sess.run(tf.argmax(hypothesis,1),feed_dict={X:mnist.test.images[r:r+1]}))

plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
#plt.savefig("test.png")
plt.show()
