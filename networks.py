from scipy.special import expit, softmax
import numpy as np


class Layer:
    def forward(self, **kwargs):
        pass

    def backward(self, input_grad):
        pass


class input(Layer):
    def __init__(self, input_shape, name="x"):
        self.output_shape = input_shape
        self.name = name

    def forward(self, **kwargs):
        self._value = kwargs[self.name]
        return self._value


class sigmoid(Layer):
    def __init__(self, previous_layer):
        self.output_shape = previous_layer.output_shape
        self.previous_layer = previous_layer

    def forward(self, **kwargs):
        self._value = expit(self.previous_layer.forward(**kwargs))
        return self._value

    def backward(self, input_grad):
        derivative = input_grad * self._value * (1 - self._value)
        self.previous_layer.backward(derivative)


class leaky_relu(Layer):
    def __init__(self, previous_layer, alpha=0.01):
        self.output_shape = previous_layer.output_shape
        self.previous_layer = previous_layer
        self.alpha = alpha

    def forward(self, **kwargs):
        input = self.previous_layer.forward(**kwargs)
        mask = (input > 0).astype("float32")
        self._value = np.where(mask, input, self.alpha * input)
        return self._value

    def backward(self, input_grad):
        self.previous_layer.backward(
            self.mask * derivative + self.alpha * (1 - self.mask) * input_grad
        )


class relu(leaky_relu):
    def __init__(self, previous_layer):
        self.output_shape = previous_layer.output_shape
        self.previous_layer = previous_layer
        self.alpha = 0


class abs(leaky_relu):
    def __init__(self, previous_layer):
        self.output_shape = previous_layer.output_shape
        self.previous_layer = previous_layer
        self.alpha = -1


class dense(Layer):
    def __init__(self, previous_layer, n_units):
        assert len(previous_layer.output_shape) == 2
        self.output_shape = (previous_layer.output_shape[0], n_units)
        self.previous_layer = previous_layer
        self.W = np.random.randn(previous_layer.output_shape[1], n_units)
        self.b = np.random.randn(n_units)

    def forward(self, **kwargs):
        self._value = (
            self.previous_layer.forward(**kwargs).dot(self.W) + self.b
        )
        return self._value

    def backward(self, input_grad):
        self._W_gradient = np.einsum(
            "nk,nd->ndk", input_grad, self.previous_layer._value
        )
        self._b_gradient = input_grad
        self.previous_layer.backward(input_grad.dot(self.W.T))


class softmax(Layer):
    def __init__(self, previous_layer):
        self.output_shape = previous_layer.output_shape
        self.previous_layer = previous_layer

    def forward(self, **kwargs):
        self._value = softmax(self.previous_layer.forward(**kwargs))
        return self._value

    def backward(self, input_grad):
        # First we create for each example feature vector, it's outer product with itself
        # ( p1^2  p1*p2  p1*p3 .... )
        # ( p2*p1 p2^2   p2*p3 .... )
        # ( ...                     )
        tensor1 = np.einsum(
            "ij,ik->ijk", self._value, self._value
        )  # (m, n, n)
        # Second we need to create an (n,n) identity of the feature vector
        # ( p1  0  0  ...  )
        # ( 0   p2 0  ...  )
        # ( ...            )
        tensor2 = np.einsum(
            "ij,jk->ijk", self._value, np.eye(n, n)
        )  # (m, n, n)
        # Then we need to subtract the first tensor from the second
        # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
        # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
        # ( ...                              )
        dSoftmax = tensor2 - tensor1
        # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
        derivative = np.einsum("ijk,ik->ij", dSoftmax, input_grad)  # (m, n)
        self.previous_layer.backward(derivative)


class MSE(Layer):
    def __init__(self, previous_layer):
        self.output_shape = ()
        self.previous_layer = previous_layer

    def forward(self, **kwargs):
        assert "y" in kwargs
        self._diff = self.previous_layer.forward(**kwargs) - kwargs["y"]
        axes = range(1, len(self.previous_layer.output_shape))
        self._value = ((self._diff) ** 2).mean(tuple(axes))
        return self._value

    def backward(self):
        self.previous_layer.backward(self._diff)


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
