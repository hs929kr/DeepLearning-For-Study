import tensorflow as tf
import numpy as np
tf.set_random_seed(777)#랜덤한 값을 얻을 때 다른 컴퓨터에서도 동일한 random값을 얻을 수 있도록 설계된 함수

#','로 구분된 값 텍스트파일(.csv)는 일반적으로 콤마 문자(,)가 각 텍스트 필드를 구분
#TAB문자가 일반적으로 각 텍스트 필드를 구분하는 구분된 텍스트 파일(.txt)
xy=np.loadtxt('data_for_linear_regression_csv_data_load/test.csv',delimiter=',',dtype=np.float32)#구분자=','를 기준으로 txt파일을 열어 xy로 (1차원,2차원,n차원)배열 값을 받아온다.
x_data=xy[:,0:-1]#행은 모든 범위를, 열은 첫번째부터 마지막 전까지의 데이터를 가져온다.
y_data=xy[:,[-1]]#행은 모든 범위를, 열은 마지막 데이터를 배열로 받아 가져온다.

print(x_data.shape,x_data,len(x_data))
print(y_data.shape,y_data,len(y_data))


X=tf.placeholder(tf.float32,shape=[None,3])
Y=tf.placeholder(tf.float32,shape=[None,1])

W=tf.Variable(tf.random_normal([3,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=tf.matmul(X,W)+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val,hy_val,_=sess.run([cost,hypothesis,train],feed_dict={X:x_data,Y:y_data})
    if step%10==0:
        print(step,cost_val,hy_val)

print("Your score will be", sess.run(hypothesis,feed_dict={X:[[100,70,101]]}))

print("Other score will be", sess.run(hypothesis,feed_dict={X:[[60,70,110],[90,100,80]]}))
