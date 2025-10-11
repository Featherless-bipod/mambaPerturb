import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score 
import numpy as np
import pandas as pd
import time
import modelMLP 
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# --- 1. The Custom Dataset for Multi-Modal Input ---
# This is the memory-safe way to handle large, multiple-input data.

class PerturbationMultiModalDataset(Dataset):
    """
    A custom PyTorch Dataset that provides two separate inputs for each sample:
    - The cell's gene expression vector.
    - The perturbation's positional encoding vector.
    """
    def __init__(self, expression_data, positional_data, labels):
        self.expression_data = expression_data 
        self.positional_data = positional_data 
        self.labels = labels                  

    def __len__(self):
        return self.expression_data.shape[0]

    def __getitem__(self, idx):
        # This method is called one sample at a time by the DataLoader.
        
        # Get the sparse row for one cell and convert ONLY that row to a dense tensor.
        expression_vector = torch.tensor(
            self.expression_data[idx].toarray().flatten(), dtype=torch.float32
        )
        
        # Get the positional data for that same cell.
        position_vector = torch.tensor(self.positional_data[idx], dtype=torch.float32)
        
        # Get the label for that cell.
        label_vector = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # The DataLoader expects the inputs to be grouped together.
        inputs = (expression_vector, position_vector)
        
        return inputs, label_vector


class MultiModalCrossAttentionModel(nn.Module):
    def __init__(self, n_genes, n_positional_features, n_classes, embed_dim=128, n_heads=8, dropout=0.2):
        super().__init__()
        
        self.embed_dim = embed_dim

        # --- Expression Tower (Encoder for gene expression) ---
        # A simple MLP to learn a dense representation from the sparse gene data.
        self.expression_encoder = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim) # Projects to the common embedding dimension
        )

        # --- Positional Tower (Encoder for perturbation location) ---
        # A simple linear layer to project the positional data to the same embedding space.
        self.positional_encoder = nn.Linear(n_positional_features, embed_dim)

        # --- Fusion Layer (Cross-Attention) ---
        # This is where the two modalities "talk" to each other.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=False # Expects (seq_len, batch, embed_dim)
        )

        # --- "Add & Norm" layers for stability (inspired by Transformers) ---
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # --- Final Prediction Head ---
        # An MLP that takes the fused representation and makes the final prediction.
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, expression_input, position_input):
        # expression_input shape: (batch_size, n_genes)
        # position_input shape: (batch_size, n_positional_features)
        
        # 1. Pass each input through its respective tower to get embeddings.
        expr_embedding = self.expression_encoder(expression_input) # Shape: (batch_size, embed_dim)
        pos_embedding = self.positional_encoder(position_input)   # Shape: (batch_size, embed_dim)
        
        # 2. Prepare inputs for the cross-attention layer.
        # MultiheadAttention expects shape (sequence_length, batch_size, embed_dim).
        # Since we have one vector per cell, our sequence_length is 1.
        # We use .unsqueeze(0) to add this dimension.
        query = expr_embedding.unsqueeze(0)    # Shape: (1, batch_size, embed_dim)
        key = pos_embedding.unsqueeze(0)       # Shape: (1, batch_size, embed_dim)
        value = pos_embedding.unsqueeze(0)     # Shape: (1, batch_size, embed_dim)

        # 3. Perform cross-attention.
        # The expression embedding "queries" the positional embedding for relevant info.
        attn_output, _ = self.cross_attention(query=query, key=key, value=value)
        
        # 4. Apply the "Add & Norm" step.
        # Add the original query (a residual connection) and then apply LayerNorm.
        fused_embedding = self.norm1(query + attn_output)
        
        # 5. Remove the sequence length dimension for the prediction head.
        fused_embedding = fused_embedding.squeeze(0) # Shape: (batch_size, embed_dim)
        
        # 6. Make the final prediction.
        logits = self.prediction_head(fused_embedding) # Shape: (batch_size, n_classes)
        
        return logits

# --- 3. Full Training Script ---

# This is a placeholder for where you would load your data.
# In your notebook, 'adata', 'labels_int', and 'X_pert_pos' should already be defined.
# For example:
# import scanpy as sc
# adata = sc.read_h5ad("path/to/your/data.h5ad")
# labels_int = ... # From MultiLabelBinarizer
# X_pert_pos = ... # From the per-cell positional encoding script

# --- Create instances of your data ---
# This assumes the variables adata, labels_int, and X_pert_pos exist
try:
    expression_data = adata.X
    positional_data = adata.obsm['X_pert_pos']
    
    # --- Create the full dataset object ---
    full_dataset = PerturbationMultiModalDataset(expression_data, positional_data, labels_int)

    # --- Split indices for training and validation ---
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        random_state=67
    )
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # --- Create DataLoaders ---
    batch_size = 32 # You might need to adjust this based on GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Define Hyperparameters and Instantiate Model ---
    model = MultiModalCrossAttentionModel(
        n_genes=adata.n_vars,
        n_positional_features=positional_data.shape[1],
        n_classes=labels_int.shape[1],
        embed_dim=128,
        n_heads=8,
        dropout=0.2
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    num_epochs = 50

    # --- Training and Evaluation Loop ---
    print("\nStarting training of the Multi-Modal Cross-Attention Model...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for (expression, position), labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            optimizer.zero_grad()
            
            # The model's forward pass now takes two inputs
            outputs = model(expression, position)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        all_test_preds, all_test_labels = [], []
        total_test_loss = 0
        with torch.no_grad():
            for (expression, position), labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                outputs = model(expression, position)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_test_preds.append(preds.cpu().numpy())
                all_test_labels.append(labels.cpu().numpy())
        
        avg_test_loss = total_test_loss / len(test_loader)
        val_f1 = f1_score(np.vstack(all_test_labels), np.vstack(all_test_preds), average='micro')
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_test_loss:.4f} | Val F1: {val_f1:.4f}")

except NameError:
    print("\nPlaceholder section: 'adata' object not found.")
    print("In your real notebook, this script would run the full training process.")
