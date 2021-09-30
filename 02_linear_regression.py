import tensorflow as tf
#train data1
'''
x_train=[1,2,3]
y_train=[1,2,3]
'''

#train data2
X=tf.placeholder(tf.float32,shape=[None])#placeholder means data declaration and assign value later , shape can be a constant or array shape=[None] means can be any shape [1,1] or [13,2] etc
Y=tf.placeholder(tf.float32,shape=[None])



#make tensorflow graph
#tensorflow updating variable declaration : not just a variable but for new concept of variable only used in tensorflow, charateristic : tensorflow update variable automatically
w=tf.Variable(tf.random_normal([1]),name='weight')#tf.random_normal : produce random number , [1] means shape
b=tf.Variable(tf.random_normal([1]),name='bias')

#hypothesis = x_train * w + b
hypothesis = X * w + b

#const function
#cost = tf.reduce_mean(tf.square(hypothesis - y_train)) #(reduce_mean : sigma/N) , (tf.squre : square opration)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#cost minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #declaring
train = optimizer.minimize(cost) #cost minimizing function

#launch the graph in a session
sess=tf.Session()
#sess.run(tf.global_variables_initializer())#to use global variable we should use initialize function(when we use train_x, train_y)

#training data1
'''
for step in range(2001):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(cost),sess.run(w),sess.run(b))
'''

#training data2
for step in range(2001):
    cost_val, w_val, b_val, _ =\
        sess.run([cost,w,b,train],feed_dict={X : [1,2,3], Y : [1,2,3]})#sess.run can active with only one argument but also list , if placeholder node is in graph, should add feed_dect to feed data to variable
    if step%20 == 0:
        print(step,cost_val,w_val,b_val)
