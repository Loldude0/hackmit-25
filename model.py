import torch.nn as nn

class EEGLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2,
                 num_classes=3, bidirectional=False, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1),
                            num_classes)

    def forward(self, x):
        # x: (batch, seq_len=5 channels, input_size=4 bands)
        out, _ = self.lstm(x)         # out: (batch, seq_len, hidden*dirs)
        out = out[:, -1, :]           # take last time-step
        return self.fc(out)