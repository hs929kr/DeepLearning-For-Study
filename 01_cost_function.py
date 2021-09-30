import tensorflow as tf
import matplotlib.pyplot as plt#for visualization
import numpy as np #numerical python : big multi demension calculating tool
from mpl_toolkits.mplot3d import Axes3D #for drawing 3D graph

X=[1,2,3]
Y=[1,2,3]

W=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)

hypothesis=X*W+b
cost=tf.reduce_mean(tf.square(hypothesis-Y))
sess=tf.Session()
sess.run(tf.global_variables_initializer())
W_val=[]
b_val=[]
cost_val=[]
for i in range(-20,20):
    for j in range(-20,20):
        feed_W=i
        feed_b=j
        curr_cost, curr_W, curr_b=sess.run([cost,W,b],feed_dict={W:feed_W, b:feed_b})
        cost_val.append(curr_cost)
        W_val.append(curr_W)
        b_val.append(curr_b)
        print(curr_W, curr_b, curr_cost)
cost_val=np.array(cost_val)#list to numpy.array
W_val=np.array(W_val)
b_val=np.array(b_val)

fig=plt.figure(figsize=(6,6))#display size setting by inch
ax=fig.add_subplot(236,projection='3d')#111 means 6subplot in 2x3 greed , projection means demension
ax.scatter(W_val,b_val,cost_val,marker='o',s=10,cmap='Greens')#scatter means draw graph by dot. so x,y,z means axis, marker means draw method of point, s means point size, cmap means color of point
plt.show()#display finally
