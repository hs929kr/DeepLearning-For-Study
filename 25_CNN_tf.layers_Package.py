import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

class Model : #first is dfine to make Class 
    def __init__(self,sess,name,learning_rate): #init method --> when making instance the init method activated automatically
        self.sess=sess #self --> to define itself we use first factor in method as a 'self' customarily 
        self.name=name
        self.learning_rate=learning_rate
        self._build_net()
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training=tf.placeholder(tf.bool)
            self.X=tf.placeholder(tf.float32,[None,784])
            self.Y=tf.placeholder(tf.float32,[None,10])
            self.keep_prob=tf.placeholder(tf.float32)            
            X_img=tf.reshape(self.X,[-1,28,28,1])
            
            #W1=tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))#stddev(standard deviation) : 표쥰편차
            #L1=tf.nn.conv2d(X_img,W1,strides=[1,1,1,1],padding='SAME')
            #L1=tf.nn.relu(L1)     
            #L1=tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            #L1=tf.nn.dropout(L1,keep_prob=self.keep_prob)
            conv1=tf.layers.conv2d(inputs=X_img,filters=32,kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
            pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],padding="SAME",strides=2)
            dropout1=tf.layers.dropout(inputs=pool1,rate=0.7,training=self.training)
            
            #W2=tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
            #L2=tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
            #L2=tf.nn.relu(L2)
            #L2=tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            #L2=tf.nn.dropout(L2,keep_prob=self.keep_prob)
            conv2=tf.layers.conv2d(inputs=dropout1,filters=64,kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
            pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],padding="SAME",strides=2)
            dropout2=tf.layers.dropout(inputs=pool2,rate=0.7,training=self.training)
            
            #W3=tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))
            #L3=tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding='SAME')
            #L3=tf.nn.relu(L3)
            #L3=tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            #L3=tf.nn.dropout(L3,keep_prob=self.keep_prob)
            conv3=tf.layers.conv2d(inputs=dropout2,filters=128,kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
            pool3=tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],padding="SAME",strides=2)
            dropout3=tf.layers.dropout(inputs=pool3,rate=0.7,training=self.training)

            #L3=tf.reshape(L3,[-1,4*4*128]) #to put in Fully Connected Layer, make line the variables
            flat=tf.reshape(dropout3,[-1,4*4*128])

            #W4=tf.get_variable("W3",shape=[4*4*128,625],initializer=tf.contrib.layers.xavier_initializer())
            #b4=tf.Variable(tf.random_normal([625]))
            #L4=tf.nn.relu(tf.matmul(L3,W4)+b4)
            #L4=tf.nn.dropout(L4,keep_prob=self.keep_prob)
            dense4=tf.layers.dense(inputs=flat,units=625,activation=tf.nn.relu)
            dropout4=tf.layers.dropout(inputs=dense4,rate=0.5,training=self.training)
                        
            #W5=tf.get_variable("W4",shape=[625,10],initializer=tf.contrib.layers.xavier_initializer())
            #b5=tf.Variable(tf.random_normal([10]))
            self.logits=tf.layers.dense(inputs=flat,units=10)

        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self,x_test,keep_prob=1.0):
        return self.sess.run(self.logits,feed_dict={self.X:x_test,self.keep_prob:1,self.training:0})
    def get_accuracy(self,x_test,y_test,keep_prob=1.0):
        return self.sess.run(self.accuracy,feed_dict={self.X:x_test,self.keep_prob:1,self.training:0})
    def train(self,x_data,y_data,keep_prob=0.7):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x_data,self.Y:y_data,self.keep_prob:0.7,self.training:1})    

training_epochs=20
batch_size=100
learning_rate=0.01

sess=tf.Session()
m1=Model(sess,"m1",learning_rate)
sess.run(tf.global_variables_initializer())
print('Learning Started!')
for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        c,_=m1.train(batch_xs,batch_ys)
        avg_cost+=c/total_batch
    print('Epoch:','%04d' %(epoch+1),'cost =', '{:.9f}'.format(avg_cost))
print('\nLearning Finished!\n')
