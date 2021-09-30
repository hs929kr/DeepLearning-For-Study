import tensorflow as tf
import numpy as np

x_data=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y_data=np.array([[0],[1],[1],[0]],dtype=np.float32)

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
W=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=tf.sigmoid(tf.matmul(X,W)+b)
cost=-tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
cost_sum=tf.summary.scalar("cost",cost)

predicted=tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))
summary=tf.summary.merge_all()
global_step=0
with tf.Session() as sess: # with  X as Y : start X as name of Y an when belonged sentences finished, finish X.
	sess.run(tf.global_variables_initializer())
	writer=tf.summary.FileWriter("./logs")
	writer.add_graph(sess.graph)
	for step in range(100001):
		s,_=sess.run([summary,train], feed_dict={X:x_data, Y:y_data})
		if step%100==0:
			#s=sess.run([summary],feed_dict={X:x_data,Y:y_data})
			writer.add_summary(s,global_step)
			global_step+=1
			print(step,sess.run(cost,feed_dict={X: x_data, Y: y_data}),sess.run(W))

	h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y: y_data})
	print("\nHypothesis: ",h, "\nCorrect: ",c, "\nAccuracy:",a)

