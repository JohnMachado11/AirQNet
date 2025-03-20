"""
AirQNet - Neural Network Architecture
"""

import torch


# Feedforward Neural Network
class FFN(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # Input Layer and Hidden Layer #1
            torch.nn.Linear(num_inputs, 30),
            torch.nn.BatchNorm1d(30),
            torch.nn.ReLU(),
            
            # Hidden Layer #2
            torch.nn.Linear(30, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),  # Dropout with 40%

            # Hidden Layer #3
            torch.nn.Linear(128, 30),
            torch.nn.BatchNorm1d(30),
            torch.nn.ReLU(),

            # Hidden Layer #4
            torch.nn.Linear(30, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),  # Dropout with 40%

            # Output Layer
            torch.nn.Linear(64, num_outputs)
        )
    
    def forward(self, x):
        logits = self.layers(x)
        return logits