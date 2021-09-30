import tensorflow as tf

x_data=[1,2,3]
y_data=[1,2,3]

W=tf.Variable(tf.random_normal([1]),name='weight')#try 5, -3
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

hypothesis=X*W

cost=tf.reduce_mean(tf.square(hypothesis-Y))

#minimize : gradient descent using derivative : W-=Learning_rate*derivative
learning_rate=0.1
gradient=tf.reduce_mean((W*X-Y)*X)
descent=W-learning_rate*gradient
update=W.assign(descent)#assign means substitute 'descent' to 'W'
'''
#minimize : gradient Descent Magic , differential about tf.Variable
oprimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train=optimizer.minimize(cost)
'''
'''
#same gradient and W : functions are same  
gvs=optimizer.comput_gradients(cost)
apply_gradients=optimizer.apply_gradients(gvs)
'''


sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step,sess.run(cost,feed_dict={X: x_data, Y: y_data}), sess.run(W))
