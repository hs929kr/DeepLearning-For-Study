import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import random

class Model :
    def __init__(self,sess,name,learning_rate): 
        self.sess=sess 
        self.name=name
        self.learning_rate=learning_rate
        self._build_net()
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training=tf.placeholder(tf.bool)
            self.X=tf.placeholder(tf.float32,[None,200,200])
            self.Y=tf.placeholder(tf.float32,[None,2])
            self.keep_prob=tf.placeholder(tf.float32)            
            X_img=tf.reshape(self.X,[-1,50,50,1])
            conv1=tf.layers.conv2d(inputs=X_img,filters=32,kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
            pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],padding="SAME",strides=2)
            dropout1=tf.layers.dropout(inputs=pool1,rate=self.keep_prob,training=self.training)
            #output=(?,25,15,32)
            conv2=tf.layers.conv2d(inputs=dropout1,filters=64,kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
            pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],padding="SAME",strides=2)
            dropout2=tf.layers.dropout(inputs=pool2,rate=self.keep_prob,training=self.training)
            #output(?,13,8,64)
            conv3=tf.layers.conv2d(inputs=dropout2,filters=128,kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
            pool3=tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],padding="SAME",strides=2)
            dropout3=tf.layers.dropout(inputs=pool3,rate=self.keep_prob,training=self.training)
            #output(?,7,4,128)
            flat=tf.reshape(dropout3,[-1,7*7*128])
            dense4=tf.layers.dense(inputs=flat,units=625,activation=tf.nn.relu)
            dropout4=tf.layers.dropout(inputs=dense4,rate=self.keep_prob,training=self.training)
            self.logits=tf.layers.dense(inputs=flat,units=2)
            
        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def predict(self,x_test,keep_prob=1.0,training=0):
        return self.sess.run(self.logits,feed_dict={self.X:x_test,self.keep_prob:keep_prob,self.training:training})
    def get_accuracy(self,x_test,y_test,keep_prob=1.0,training=0):
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.sess.run(self.accuracy,feed_dict={self.X:x_test,self.Y:y_test,self.keep_prob:keep_prob,self.training:training})
    def train(self,x_data,y_data,keep_prob=0.7,training=1):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x_data,self.Y:y_data,self.keep_prob:keep_prob,self.training:training})    

training_epochs=50
batch_size=100
learning_rate=0.0001

sess=tf.Session()
models=[]
num_models=7
for m in range(num_models):
    models.append(Model(sess,"model"+str(m),learning_rate))
sess.run(tf.global_variables_initializer())

#data loading
print("\ndata loading\n . . . ")
image_path=".\\blackroom"
image_path_fresh=image_path+"\\ori_fresh"
image_path_rotten=image_path+"\\ori_rotten"
file_list_fresh=os.listdir(image_path_fresh)
file_list_rotten=os.listdir(image_path_rotten)
image_fresh=[]
image_rotten=[]
for i in range(len(file_list_fresh)):
    image_fresh.append(image_path_fresh+"\\"+file_list_fresh[i])
for i in range(len(file_list_rotten)):
    image_rotten.append(image_path_rotten+"\\"+file_list_rotten[i])


train_data=np.empty((1,200,200))
train_label=np.empty((1,2))
for i in range(len(image_fresh)):
    img=Image.open(image_fresh[i]).convert("L")
    img=img.resize((200,200))
    img=np.array(img)
    train_data=np.append(train_data,[img],axis=0)
    train_label=np.append(train_label,[[1,0]],axis=0)
    if((i+1)%100==0):
        print("in fresh image ",len(image_fresh),", ",i+1," preprocessed.")

for i in range(len(image_rotten)):
    img=Image.open(image_rotten[i]).convert("L")
    img=img.resize((200,200))
    img=np.array(img)
    train_data=np.append(train_data,[img],axis=0)
    train_label=np.append(train_label,[[0,1]],axis=0)
    if((i+1)%100==0):
        print("in rotten image ",len(image_rotten),", ",i+1," preprocessed.")

train_data=np.delete(train_data,[0,0],axis=0)
train_label=np.delete(train_label,[0,0],axis=0)

#input data random shuffling
idx=np.arange(train_data.shape[0])
np.random.shuffle(idx)
train_data=train_data[idx]
train_label=train_label[idx]
train_num=len(train_data)
test_data=train_data
test_label=train_label

#Learning stage
print('\nLearning Started!')
for epoch in range(training_epochs):
    avg_cost_list=np.zeros(len(models))
    total_batch=int(train_num/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys=train_data[batch_size*i:batch_size*(i+1)],train_label[batch_size*i:batch_size*(i+1)]
        for m_idx,m in enumerate(models):#enumerate : as a python builtin function, return index and value in list
            c,_=m.train(batch_xs,batch_ys)
            avg_cost_list[m_idx]+=c/total_batch

    print('Epoch:','%04d' %(epoch+1),'cost =',avg_cost_list)
    '''if(epoch==30):
        for m_idx,m in enumerate(models):
            m.learning_rate=0.0001'''
print('\nLearning Finished!\n')

#detection stage
test_size=len(test_label)
test_batch=int(test_size/batch_size)
predictions=np.zeros(test_batch*batch_size*2).reshape(test_batch*batch_size,2)

for i in range(test_batch):
    batch_xs,batch_ys=test_data[batch_size*i:batch_size*(i+1)],test_label[batch_size*i:batch_size*(i+1)]
    for m_idx,m in enumerate(models):
        print('epoch : ',i+1,'model : ',m_idx+1,'Accuracy:',m.get_accuracy(batch_xs,batch_ys))
        p=m.predict(batch_xs)
        predictions[batch_size*i:batch_size*(i+1)]+=p
ensemble_correct_prediction=tf.equal(tf.argmax(predictions,1),tf.argmax(test_label[0:total_batch*batch_size],1))
ensemble_accuracy=tf.reduce_mean(tf.cast(ensemble_correct_prediction,tf.float32))
print('Ensemble accuracy:',sess.run(ensemble_accuracy))
