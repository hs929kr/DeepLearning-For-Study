import tensorflow as tf
import numpy as np

xy=np.genfromtxt('./data_for_linear_regression_csv_data_load/zoo/zoo.csv',delimiter=',',dtype=np.float32)
x_data=xy[1:,1:-1]
y_data=xy[1:,[-1]]

nb_classes=7

X=tf.placeholder(tf.float32,[None,16])
Y=tf.placeholder(tf.int32,[None,1])

Y_one_hot=tf.one_hot(Y,nb_classes) #tf.one_hot --> label to one_hot
#one_hot function --> input indices is rank N, the output will have rank N+1
#[[0],[3]]-->one_hot_func-->[[[1,0,0,0,0,0]],[[0,1,0,0,0,0,0]]] : shape=[?,1,7]
Y_one_hot=tf.reshape(Y_one_hot,[-1,nb_classes])
#reshape --> change shape to what we want (-1 mean auto)

W=tf.Variable(tf.random_normal([16,nb_classes]),name='weight')
b=tf.Variable(tf.random_normal([nb_classes]),name='bias')

logits=tf.matmul(X,W)+b
hypothesis=tf.nn.softmax(logits)

cost_i=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y_one_hot)
cost=tf.reduce_mean(cost_i)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction=tf.argmax(hypothesis,1)
#argmax : print biggest data index in matrix
correct_prediction=tf.equal(prediction,tf.argmax(Y_one_hot,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(2000):
		sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
		if step%100==0:
			loss,acc=sess.run([cost,accuracy],feed_dict={X:x_data,Y:y_data})
			print("Step: {:5}\tLoss: {:.3f}".format(step,loss,acc))

	pred=sess.run(prediction,feed_dict={X:x_data}) #zip : match elements in each matrix data
	for p,y in zip(pred,y_data.flatten()): #flatten : make any matrix to shape = [-1]
		print("[{}] Prediction: {} True Y: {}".format(p==int(y),p,int(y)))
