from lib.layer.rbf_layer import RBFLayer
from src.util.fun import (
    l_norm,
    rbf_gaussian,
    rbf_inverse_multiquadric,
    rbf_inverse_quadratic,
    rbf_linear,
    rbf_multiquadric,
)


from torch import nn


class RbnModel(nn.Module):

    activations = {
        "gaussian": rbf_gaussian,
        "linear": rbf_linear,
        "multiquadric": rbf_multiquadric,
        "inverse_quadratic": rbf_inverse_quadratic,
        "inverse_multiquadric": rbf_inverse_multiquadric,
    }

    def __init__(self, activation, input_size, num_kernels, output_size):
        super(RbnModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.rbf = RBFLayer(
            in_features_dim=self.input_size,  # input features dimensionality
            num_kernels=num_kernels,  # number of kernels
            out_features_dim=self.output_size,  # output features dimensionality
            radial_function=self.activations[activation],  # radial basis function used
            norm_function=l_norm,
        )  # l_norm defines the \ell norm

    def forward(self, x):
        return self.rbf(x)
