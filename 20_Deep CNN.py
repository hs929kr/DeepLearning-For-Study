import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
import matplotlib.pyplot as plt
import random

#set global variables
training_epochs=20
batch_size=100
learning_rate=0.01
keep_prob=tf.placeholder(tf.float32) #probability

#set input Value
X=tf.placeholder(tf.float32,[None,784])
X_img=tf.reshape(X,[-1,28,28,1])
print(X.shape)
print(X_img.shape)
Y=tf.placeholder(tf.float32,[None,10])
print(Y.shape)

#set Layer1 Variable : input-28*28, output-14*14
W1=tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))#stddev(standard deviation) : 표쥰편차
L1=tf.nn.conv2d(X_img,W1,strides=[1,1,1,1],padding='SAME')
L1=tf.nn.relu(L1)
L1=tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L1=tf.nn.dropout(L1,keep_prob=keep_prob)

#set Layer2 Variable : input-14*14, output-8*8
W2=tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
L2=tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
L2=tf.nn.relu(L2)
L2=tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L2=tf.nn.dropout(L2,keep_prob=keep_prob)

#set Layer3 Variable: input-8*8, output-4*4
W3=tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))
L3=tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding='SAME')
L3=tf.nn.relu(L3)
L3=tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L3=tf.nn.dropout(L3,keep_prob=keep_prob)

#Fully Connected Layer
L3=tf.reshape(L3,[-1,4*4*128]) #to put in Fully Connected Layer, make line the variables
W4=tf.get_variable("W3",shape=[4*4*128,625],initializer=tf.contrib.layers.xavier_initializer())
b4=tf.Variable(tf.random_normal([625]))
L4=tf.nn.relu(tf.matmul(L3,W4)+b4)
L4=tf.nn.dropout(L4,keep_prob=keep_prob)

W5=tf.get_variable("W4",shape=[625,10],initializer=tf.contrib.layers.xavier_initializer())
b5=tf.Variable(tf.random_normal([10]))
hypothesis=tf.matmul(L4,W5)+b5

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
                feed_dict={X:batch_xs,Y:batch_ys,keep_prob:0.7}
                c,_=sess.run([cost,optimizer],feed_dict=feed_dict)
                avg_cost+=c/total_batch
        print('Epoch:','%04d' %(epoch+1),'cost =', '{:.9f}'.format(avg_cost))
print('\nLearning Finished!\n')

correct_prediction=tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('Accuracy:',sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.lables,keep_prob:1.0}))