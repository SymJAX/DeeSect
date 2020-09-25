import numpy as np
import layers
import networks
import surface


class MLP(networks.MLP):
    def preactivation_mapping(self, x):
        self.layers[-1].forward(x=x)
        preactivations = []
        for layer in self.layers:
            if isinstance(layer, layers.dense):
                preactivations.append(layer._value.reshape((x.shape[0], -1)))
        return np.concatenate(preactivations, 1)

    def code_mapping(self, x):
        self.layers[-1].forward(x=x)
        codes = []
        for layer in self.layers:
            if isinstance(layer, layers.leaky_relu):
                codes.append(layer._mask)
        codes = [p.reshape((x.shape[0], -1)) for p in codes]
        return np.concatenate(codes, 1)

    def output_mapping(self, x):
        return self.layers[-1].forward(x=x)


np.random.seed(8)
surface.pretty_onelayer_partition(
    np.random.rand(3, 2) * 4 - 2,
    np.random.randn(3) * 0.5,
    -0.01,
    name="one_layer_PD.pdf",
)
K = 6
np.random.seed(1)
surface.pretty_onelayer_partition(
    np.random.rand(K, 2) * 1 - 0.5,
    np.random.randn(K) * 0.0,
    -0.1,
    with_power_diagram=True,
    name="one_layer_biasless.pdf",
)

np.random.seed(1)
W = np.random.randn(K, 2) * 0.4
for k in range(K):
    W[k, int(np.random.rand() < 0.5)] = 0
surface.pretty_onelayer_partition(
    W,
    np.random.rand(K) * 2 - 1,
    -0.1,
    with_power_diagram=True,
    name="one_layer_l0.pdf",
)

np.random.seed(2)
theta = np.linspace(-np.pi, np.pi, 100)
W = np.stack([np.cos(theta), np.sin(theta)], 1)
surface.pretty_onelayer_partition(
    W,
    np.ones(100) * 0.5,
    -0.01,
    with_power_diagram=False,
    name="one_layer_circle.pdf",
)


np.random.seed(11)
model = MLP([layers.input((10, 2))], [8, 8], alpha=0.3)
model.layers.append(layers.dense(model.layers[-1], 2))

surface.pretty_plot(
    model.preactivation_mapping,
    model.output_mapping,
    model.code_mapping,
    n_samples_x=200,
    n_samples_y=200,
    name_input_space="input_space_2d.pdf",
    name_output_space="output_space_2d.pdf",
)

np.random.seed(11)
model = MLP([layers.input((10, 2))], [8, 8], alpha=0.3)
model.layers.append(layers.dense(model.layers[-1], 2))
from scipy.stats import multivariate_normal

var = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
surface.pretty_plot(
    model.preactivation_mapping,
    model.output_mapping,
    model.code_mapping,
    n_samples_x=200,
    n_samples_y=200,
    color_mapping=var.pdf,
    name_input_space="input_space_2d_g.pdf",
    name_output_space="output_space_2d_g.pdf",
)


# np.random.seed(12)
# model = MLP([layers.input((10, 2))], [8, 8], alpha=0.3)
# model.layers.append(layers.dense(model.layers[-1], 3))

# surface.pretty_plot(
#     model.preactivation_mapping,
#     model.output_mapping,
#     model.code_mapping,
#     n_samples_x=200,
#     n_samples_y=200,
#     name_input_space="input_space_3d.pdf",
#     name_output_space="output_space_3d.pdf",
# )
