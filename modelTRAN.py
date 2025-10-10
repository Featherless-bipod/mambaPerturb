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
import torch
import math
from torch import nn

class TransMLP(nn.Module):
    def __init__(self,input_dim,num_classes,num_heads,embed_dim,hidden_dims, dropout_p):
        super(TransMLP, self).__init__()

        self.norm1 = nn.LayerNorm(input_dim) 
        self.mha = nn.MultiheadAttention(embed_dim,num_heads,dropout_p)
        hidden_dim = hidden_dims[0]
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.output_layer = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        attended_x = self.mha(x)
        x = self.norm1(x + attended_x)

        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output) 

        output = self.output_layer(x)
        return output


