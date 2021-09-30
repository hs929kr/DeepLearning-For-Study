import tensorflow as tf

x1_data=[73.,93.,89.,96.,73.]
x2_data=[80.,88.,91.,98.,66.]
x3_data=[75.,93.,90.,100.,70.]
y_data=[152.,185.,180.,196.,142.]

x1=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)
x3=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

w1=tf.Variable(tf.random_normal([1]),name='weight1')
w2=tf.Variable(tf.random_normal([1]),name='weight2')
w3=tf.Variable(tf.random_normal([1]),name='weight3')
b=tf.Variable(tf.random_normal([1]),name='bias')
hypothesis=x1*w1+x2*w2+x3*w3+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(200000001):
    W1,W2,W3,B,cost_val,hy_val,_=sess.run([w1.initialized_value(),w2.initialized_value(),w3.initialized_value(),b.initialized_value(),cost,hypothesis,train],feed_dict={x1:x1_data, x2:x2_data, x3:x3_data,Y:y_data})
    if step%10==0:
        print(str(step),'\nvarables:',W1,W2,W3,B,"\nCost:",cost_val,"\nPrediction:",hy_val,'\n')
