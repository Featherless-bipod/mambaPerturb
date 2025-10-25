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
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import functions as fun


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
        self.expression_encoder = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim) # Projects to the common embedding dimension
        )

        # --- Positional Tower (Encoder for perturbation location) ---
        self.positional_encoder = nn.Linear(n_positional_features, embed_dim)

        # --- Fusion Layer (Cross-Attention) ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=False # Expects (seq_len, batch, embed_dim)
        )

        # --- "Add & Norm" layers for stability (inspired by Transformers) ---
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # --- Final Prediction Head ---
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
        query = expr_embedding.unsqueeze(0)    # Shape: (1, batch_size, embed_dim)
        key = pos_embedding.unsqueeze(0)       # Shape: (1, batch_size, embed_dim)
        value = pos_embedding.unsqueeze(0)     # Shape: (1, batch_size, embed_dim)

        # 3. Perform cross-attention.
        attn_output, _ = self.cross_attention(query=query, key=key, value=value)
        
        # 4. Apply the "Add & Norm" step.
        fused_embedding = self.norm1(query + attn_output)
        
        # 5. Remove the sequence length dimension for the prediction head.
        fused_embedding = fused_embedding.squeeze(0) # Shape: (batch_size, embed_dim)
        
        # 6. Make the final prediction.
        logits = self.prediction_head(fused_embedding) # Shape: (batch_size, n_classes)
        
        return logits
    
def train(model, train_loader, test_loader, optimizer, criterion, num_epochs):
    # --- 1. Initialization for Plotting ---
    # Create lists to store the history of metrics
    train_losses,val_losses = [],[]
    train_f1_scores,val_f1_scores = [], []
    all_acc1_scores, all_acc5_scores = [],[]

    # Create a figure and axes for the plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    # Initially display the empty figure
    display(fig)

    print("\nStarting training with live plotting...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        # Use tqdm for a progress bar on the training data
        for (expression, position), labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            optimizer.zero_grad()
            # The model's forward pass now takes two inputs
            outputs = model(expression, position)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Evaluation Phase ---
        model.eval()
        all_val_preds, all_val_labels = [], []
        all_train_preds, all_train_labels = [], []
        all_acc1, all_acc5 = [],[]
        total_val_loss = 0
        
        with torch.no_grad():
            # Get validation metrics
            for (expression, position), labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                outputs = model(expression, position)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_val_preds.append(preds.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

                '''if labels.dim() == 2:
                    labels = labels.argmax(dim=1)

                acc1 = topk_accuracy(outputs,labels, k=1)
                acc5 = topk_accuracy(outputs,labels, k=5)
                
                all_acc1.append(acc1)
                all_acc5.append(acc5)'''


            # Also get training metrics for comparison (important for diagnosing overfitting)
            for (expression, position), labels in train_loader:
                outputs = model(expression, position)

                '''if labels.dim() == 2:
                    labels = labels.argmax(dim=1)
                '''
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_train_preds.append(preds.cpu().numpy())
                all_train_labels.append(labels.cpu().numpy())



        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)


        train_f1 = f1_score(np.vstack(all_train_labels), np.vstack(all_train_preds), average='micro')
        val_f1 = f1_score(np.vstack(all_val_labels), np.vstack(all_val_preds), average='micro')

        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)
        '''all_acc1_scores.append(acc1)
        all_acc5_scores.append(acc5)'''

        fun.plot(fig, ax1, ax2, ax3, train_losses, val_losses, train_f1_scores, val_f1_scores)

        # Print a summary for the current epoch
        print(f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
        f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} ")

    # --- 3. Cleanup after the loop is done ---
    plt.close(fig)
    print(f"\nTraining Complete. Final Validation F1 Score: {val_f1_scores[-1]:.4f}")