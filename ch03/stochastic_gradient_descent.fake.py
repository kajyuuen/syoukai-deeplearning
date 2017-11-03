# 確率的勾配降下法の疑似コード

for epoch in range(epochs):
    shuffle(data)
    for datum in data:
        params_grad = evaluate_gradient(error_function, params, datum)
        params -= learning_rate * params_grad
