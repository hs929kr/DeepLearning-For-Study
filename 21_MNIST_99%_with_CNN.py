import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
import matplotlib.pyplot as plt
import random

#set global variables
training_epochs=20
batch_size=100
learning_rate=0.01

#set input Value
X=tf.placeholder(tf.float32,[None,784])
X_img=tf.reshape(X,[-1,28,28,1])
#-1 mean no care with how many input images
print(X.shape)
print(X_img.shape)
Y=tf.placeholder(tf.float32,[None,10])
print(Y.shape)

#set Layer1 Variable
W1=tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))#stddev(standard deviation) : 표쥰편차
#make filter 3*3, grey scale,input depth : 1, ouput depth for each input(number of output filters) : 32
#if input num : 1, to make Conv --> (?,28,28,32), Pool --> (?,14,14,32)
L1=tf.nn.conv2d(X_img,W1,strides=[1,1,1,1],padding='SAME')
#strides : [base=1, height_moving, width_moving, base=1]
#padding : SAME --> if stride is 1*1 then input size=ouput_size and change by stride, valid --> no padding and if pixels are scarce, drop pixel in input
L1=tf.nn.relu(L1)
L1=tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#ksize : [number, height, width, channel]
#if stride : 1*1 then output became 28*28 because padding : same, but stride : 2*2, so output becate 14*14

#set Layer2 Variable
W2=tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
#input depth : 32, output depth for each input : 64
#if input num : 1 to make Conv --> (?,14,14,64), Pool --> (?,7,7,64)
L2=tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
L2=tf.nn.relu(L2)
L2=tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#Fully Connected Layer
L2=tf.reshape(L2,[-1,7*7*64]) #to put in Fully Connected Layer, make line the variables
W3=tf.get_variable("W2",shape=[7*7*64,10],initializer=tf.contrib.layers.xavier_initializer())
b=tf.Variable(tf.random_normal([10]))
hypothesis=tf.matmul(L2,W3)+b
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#initialize
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#train my model
print("\nLearning started. It takes sometime.\n")
for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
                batch_xs,batch_ys=mnist.train.next_batch(batch_size)
                feed_dict={X:batch_xs,Y:batch_ys}
                c,_=sess.run([cost,optimizer],feed_dict=feed_dict)
                avg_cost+=c/total_batch
        print('Epoch:','%04d' %(epoch+1),'cost =', '{:.9f}'.format(avg_cost))
print('\nLearning Finished!\n')

correct_prediction=tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('Accuracy:',sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.lables}))