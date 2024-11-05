import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super(Model, self).__init__()
        self.num_layers = n_layers
        self.hidden_size = n_hidden
        self.rnn = nn.GRU(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        output, h = self.rnn(x, h)  # output shape: (N, L, D * H_out), h_n shape: (D * num_layers, N, H_out)
        output = self.output_layer(output)  # shape: (N, L, 1)
        output = output.squeeze(-1)  # shape: (N, L)
        return output, h

if __name__ == '__main__':
    x = torch.zeros((100, 2, 1)).float()
    model = Model(25)
    print(model(x))