import torch
import torch.nn as nn


class TextNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, hc2fc, dropout):
        super(TextNet, self).__init__()

        # Network building blocks
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hc2fc),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hc2fc, hc2fc//2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hc2fc//2, output_size),
            nn.Softmax(dim=2)
        )

        # Configure hidden layer
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_hidden(self):
        # Create the initial hidden state
        return torch.zeros(self.num_layers, 1, self.hidden_size)

    def forward(self, x, hn):
        # Run forward propagation
        p, hn = self.rnn(x, hn)
        hn.detach_()
        p = self.fc(p)
        return p, hn
