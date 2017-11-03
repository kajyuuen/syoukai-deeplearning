# ミニバッチ勾配降下法の疑似コード
# M=1のとき確率的勾配降下法

for epoch in range(epochs):
    shuffle(data)
    batches = get_batches(data, batch_size=M)
    for batch in batches:
        params_grad = evaluate_gradient(error_function, params, batch)
        params -= learning_rate * params_grad
