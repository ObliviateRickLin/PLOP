import torch.nn as nn

class EgoPastEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=1):
        super(EgoPastEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Taking the output from the last time step
        out = out[:, -1, :]
        return out
    
class NbrsPastEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=1):
        super(NbrsPastEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch_size, N, sequence_length, input_size)
        batch_size, N, seq_len, input_size = x.size()
        x = x.view(batch_size * N, seq_len, input_size)  # Reshape for LSTM
        out, _ = self.lstm(x)
        # Taking the output from the last time step and reshaping back
        out = out[:, -1, :].view(batch_size, N, -1)
        return out
