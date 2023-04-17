import torch
import torch.nn as nn

class TransformerAudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x
