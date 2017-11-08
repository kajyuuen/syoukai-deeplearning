import numpy as np
import tensorflow as tf

# パラメータ
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

# モデルの構築
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid(tf.matmul(x, w) + b)

# 交差エントロピー誤差関数
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

# 勾配降下法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 学習が正しいかどうか
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# 学習用のデータ
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# モデルの初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 学習
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
        })

# 出力
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
print(classified)

prob = y.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
print(prob)

print("w:{}".format(sess.run(w)))
print("b:{}".format(sess.run(b)))
