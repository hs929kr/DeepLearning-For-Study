import tensorflow as tf
import numpy as np

xy=np.loadtxt('./data_for_linear_regression_csv_data_load/classifying_diabetes.csv',delimiter=',',dtype=np.float32)
#data : Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

X=tf.placeholder(tf.float32,shape=[None,8])
Y=tf.placeholder(tf.float32,shape=[None,1])

W=tf.Variable(tf.random_normal([8,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=tf.sigmoid((tf.matmul(X,W)+b)/30)
cost=-tf.reduce_mean(Y*tf.log(hypothesis) +(1-Y)*tf.log(1-hypothesis))
train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted=tf.cast(hypothesis>0.5,dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))#percentage

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(1000000000000000000000000001):
		W_val,b_val,accuracy_val,hypothesis_val,cost_val,_=sess.run([W,b,accuracy,hypothesis,cost,train],feed_dict={X:x_data,Y:y_data})
		if step%200==0:
			print(W_val,"\n",b_val)
			print("step : %d , cost : %f, accuracy : %f"%(step,cost_val,accuracy_val))
	print(W,b)
