from torch import nn


class IrisMlp(nn.Module):

    activations = {"relu": nn.ReLU(), "elu": nn.ELU()}

    def __init__(self, activation, hidden_layer_size):
        super(IrisMlp, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(4, hidden_layer_size), self.activations[activation]
        )
        self.classifier = nn.Linear(hidden_layer_size, 3)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.classifier(x)
        return x


class IrisRbn(nn.Module):
    pass
