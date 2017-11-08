import numpy as np
import tensorflow as tf

# XORのデータ
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# XORゲートの実装

## 入力層 
x = tf.placeholder(tf.float32, shape=[None, 2])
## 出力層
t = tf.placeholder(tf.float32, shape=[None, 1])
## 入力層-隠れ層
W = tf.Variable(tf.truncated_normal([2, 2]))
b = tf.Variable(tf.zeros([2]))
h = tf.nn.sigmoid(tf.matmul(x, W) + b)
## 隠れ層-出力層
V = tf.Variable(tf.truncated_normal([2, 1]))
c = tf.Variable(tf.zeros([2]))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

# 誤差関数
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

# 確率的勾配降下法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# ニューロン発火条件
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# 学習
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(4000):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })
    if epoch % 1000 == 0:
        print("epoch{}".format(epoch))

# 評価
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
print(classified)

prob = y.eval(session=sess, feed_dict={
    x: X
})
print(prob)
