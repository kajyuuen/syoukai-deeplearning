import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# モデルの定義
model = Sequential([
    Dense(input_dim=2, units=1),
    Activation('sigmoid')
])

# 確率的勾配法
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# 学習
model.fit(X, Y, epochs=200, batch_size=1)

# 出力
classes = model.predict_classes(X, batch_size=1)
print(classes)

prob = model.predict_proba(X, batch_size=1)
print(prob)
