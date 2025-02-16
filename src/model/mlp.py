from torch import nn


class MlpModel(nn.Module):

    activations = {"relu": nn.ReLU(), "elu": nn.ELU(), "identity": nn.Identity()}

    def __init__(self, activation, input_size, hidden_layer_size, output_size):
        super(MlpModel, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_units = nn.Sequential(
            nn.Linear(self.input_size, hidden_layer_size), self.activations[activation]
        )
        self.classifier = nn.Linear(hidden_layer_size, self.output_size)

    def forward(self, x):
        x = self.hidden_units(x)
        x = self.classifier(x)
        return x
