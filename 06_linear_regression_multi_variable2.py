import tensorflow as tf

x_data=[[73,80,75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]]
y_data=[[152.],[185.],[180.],[196.],[142.]]

X=tf.placeholder(tf.float32,shape=[None,3])
Y=tf.placeholder(tf.float32,shape=[None,1])

w=tf.Variable(tf.random_normal([3,1]),name='weight')
b=tf.Variable(tf.random_normal([3]),name='bias')
hypothesis=tf.matmul(X,w)+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(200000001):
    W,B,cost_val,hy_val,_=sess.run([w.initialized_value(),b.initialized_value(),cost,hypothesis,train],feed_dict={X:x_data,Y:y_data})
    if step%10==0:
        print(str(step),'\nvarables:',W,B,"\nCost:",cost_val,"\nPrediction:",hy_val,'\n')
