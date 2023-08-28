import torch.nn as nn

class EgoPastLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=1):
        super(EgoPastLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Taking the output from the last time step
        out = out[:, -1, :]
        return out
