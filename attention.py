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
import mlpModel 
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
        
        # --- Attention Layer ---
        self.attention = FeatureAttention(input_dim, attention_dim)
        
        # --- Dynamic MLP Layers ---
        
        mlp_layers = []
        in_features = input_dim
        self.initial_layer = nn.Linear(in_features, in_features)
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(in_features, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_p))
            in_features = hidden_dim
            
        mlp_layers.append(nn.Linear(in_features, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # Pass data through attention first
        mod_x = self.initial_layer(x)
        attended_x = self.attention(mod_x)
        # Then pass the attended features to the MLP
        output = self.mlp(attended_x)
        return output