import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size1=219, hidden_size2=438,hidden_size3 = 219, hidden_size4 = 128):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),  # Activation function
            nn.Dropout(0.3), # Dropout for regularization
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),  # Activation function
            nn.Dropout(0.3), # Dropout for regularization
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),  # Activation function
            nn.Dropout(0.3), # Dropout for regularization)
            nn.Linear(hidden_size3, hidden_size4),
            nn.ReLU(),  # Activation function
            nn.Dropout(0.3), # Dropout for regularization)
            nn.Linear(hidden_size4, num_classes) # Output layer
        )

    def forward(self, x):
        return self.layers(x)