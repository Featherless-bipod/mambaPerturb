import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score 
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class FeatureAttention(nn.Module):
    """Calculates attention weights for input features."""
    def __init__(self, input_dim, attention_dim):
        super(FeatureAttention, self).__init__()
        # A small network to compute attention scores
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, input_dim)
        )

    def forward(self, x):
        attention_logits = self.attention_net(x)
        attention_weights = F.softmax(attention_logits, dim=1)
        attended_features = x * attention_weights
        output = attended_features + x 
        return output

class AttentiveMLP(nn.Module):
    def __init__(self, input_dim, num_classes, attention_dim, hidden_dims, dropout_p):
        super(AttentiveMLP, self).__init__()

        # --- Part 1: Attention Block ---
        self.attention = FeatureAttention(input_dim, attention_dim)
        # LayerNorm should be initialized with the size of the features it will normalize
        self.norm1 = nn.LayerNorm(input_dim) 

        # --- Part 2: FeedForward Block ---
        # This block processes the features but its output size must match its input size
        # so the residual connection can be added.
        # Assuming hidden_dims = [219] from your Optuna search
        hidden_dim = hidden_dims[0]
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, input_dim) # Project BACK to input_dim
        )
        self.norm2 = nn.LayerNorm(input_dim)

        # --- Part 3: Final Classification Head ---
        # A single, final linear layer to make the prediction.
        self.output_layer = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_dim)

        # 1. Attention block with a residual connection
        attended_x = self.attention(x)
        x = self.norm1(x + attended_x) # Add original input and then normalize

        # 2. Feedforward block with another residual connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output) # Add the input of the block to its output, then normalize

        # 3. Final classification
        output = self.output_layer(x)
        
        return output