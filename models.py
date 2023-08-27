import torch.nn as nn

# Artificial Neural Network (ANN) Model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

# Convolutional Neural Network (CNN) Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        logits = self.linear_stack(x)
        return logits

# Recurrent Neural Network (RNN) Model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=28, hidden_size=128, num_layers=2, batch_first=True)
        self.linear = nn.Linear(128, 10)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out[:, -1, :])
        return out

# Transformer Model (using only the encoder part)
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=28, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.linear = nn.Linear(28, 10)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for transformer
        out = self.transformer_encoder(x)
        out = self.linear(out[:, -1, :])
        return out