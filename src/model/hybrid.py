from src.model.mlp import MlpModel
from src.model.rbn import RbnModel


from torch import nn


class HybridMplRbn(nn.Module):

    def __init__(self, mlp: MlpModel, rbn: RbnModel):
        super(HybridMplRbn, self).__init__()

        self.mlp = mlp
        self.mlp.classifier = nn.Identity()
        self.adapter = nn.Linear(mlp.hidden_layer_size, rbn.input_size)
        self.rbn = rbn

    def forward(self, x):
        x = self.mlp(x)
        x = self.adapter(x)
        x = self.rbn(x)
        return x
