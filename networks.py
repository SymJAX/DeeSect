import layers


class MLP:
    def __init__(self, previous_layers, units, alpha=0.1):
        self.layers = previous_layers
        for layer in range(len(units)):
            self.layers.append(layers.dense(self.layers[-1], units[layer]))
            self.layers.append(layers.leaky_relu(self.layers[-1], alpha=alpha))


if __name__ == "__main__":
    l0 = input((100, 800))
    l1 = dense(l0, 1000)
    l2 = sigmoid(l1)
    l3 = dense(l2, 1000)
    l4 = sigmoid(l3)
    l5 = dense(l4, 1)
    loss = MSE(l3)

    for i in range(100):
        print(
            loss.forward(
                x=np.random.randn(100, 800), y=np.random.randn(100, 1)
            ).mean()
        )
        loss.backward()
        l1.W -= 0.001 * l1._W_gradient.mean(0)
        l1.b -= 0.001 * l1._b_gradient.mean(0)
        l3.W -= 0.001 * l3._W_gradient.mean(0)
        l3.b -= 0.001 * l3._b_gradient.mean(0)
