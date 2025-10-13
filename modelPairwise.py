import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer as mlb
from tqdm import tqdm
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import random as sparse_random

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time

def prepare_graph_data(adata, mlb, positional_encodings, parsed_labels):
    """
    Converts AnnData matrices into a single PyTorch Geometric Data object.
    This is where the merging of different data modalities happens.
    """
    print("--- Step 1: Preparing graph data ---")
    
    # --- Input A: Node Features (The Phenotype) ---
    # We use the PCA data as the initial description of each cell.
    node_features = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32)
    print(f"Node features created with shape: {node_features.shape}")

    # --- Graph Structure ---
    # Convert the SciPy sparse connectivities matrix to PyG's edge_index format.
    # edge_index has shape [2, num_edges] and lists all connections.
    connectivities_matrix = adata.obsp['connectivities']
    edge_index, edge_weight_conn = from_scipy_sparse_matrix(connectivities_matrix)
    print(f"Graph structure created with {edge_index.shape[1]} edges.")

    # --- Input B: Edge Features (Merging Graph + Position) ---
    # This is where we build the "pair-wise feature graph".
    print("Constructing edge features by merging positional, distance, and connectivity data...")
    
    # Get the source and target node for each edge
    source_nodes, target_nodes = edge_index[0], edge_index[1]
    
    # Get the positional encodings for all cells
    pert_pos_features = positional_encodings
    
    # Look up the positional features for the source and target of each edge
    pos_i = pert_pos_features[source_nodes]
    pos_j = pert_pos_features[target_nodes]
    
    # Get the distance values for each edge from the distances matrix
    distances_matrix = adata.obsp['distances']
    dist_ij = distances_matrix[source_nodes.tolist(), target_nodes.tolist()].A1
    dist_ij = torch.tensor(dist_ij, dtype=torch.float32).unsqueeze(1)
    
    # Get the connectivity values for each edge (we already have this from the conversion)
    conn_ij = edge_weight_conn.unsqueeze(1)
    
    # Concatenate everything to create the final edge feature matrix
    edge_attr = torch.cat([
        torch.tensor(pos_i, dtype=torch.float32),
        torch.tensor(pos_j, dtype=torch.float32),
        dist_ij,
        conn_ij
    ], dim=1)
    print(f"Edge features created with shape: {edge_attr.shape}")
    
    # --- Labels ---
    labels = torch.tensor(mlb.transform(parsed_labels), dtype=torch.float32)

    # --- Create the final PyG Data object ---
    graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    
    # Create masks for training, validation, and testing
    num_nodes = adata.n_obs
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_end = int(num_nodes * 0.7)
    val_end = int(num_nodes * 0.85)
    
    graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    graph_data.train_mask[indices[:train_end]] = True
    graph_data.val_mask[indices[train_end:val_end]] = True
    graph_data.test_mask[indices[val_end:]] = True
    
    print("Graph data preparation complete.")
    return graph_data

# --- 2. Model Definition: The Graph Attention Network (GAT) ---

class EdgeConditionedGAT(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels, heads=8):
        super().__init__()
        # GATv2Conv is a powerful version that can directly use edge features.
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=0.6)
        
        # A final MLP head for classification
        self.classifier = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # The GAT uses the edge and phenotype features to learn node embeddings
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        
        # Final prediction based on the learned embeddings
        output = self.classifier(x)
        return output

# --- 3. Training and Evaluation Loop ---

def run_training_with_plotting(graph_data, model, optimizer, criterion, num_epochs=200):
    """
    Trains the GNN model and displays live plots of loss and F1 score.
    """
    print("\n--- Starting GAT model training with live plotting ---")
    
    # Initialization for plotting
    train_losses, val_losses = [], []
    train_f1_scores, val_f1_scores = [], []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    display(fig)

    for epoch in range(num_epochs):
        # --- Training Step ---
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        train_loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        # --- Evaluation Step ---
        model.eval()
        with torch.no_grad():
            # Re-run forward pass for evaluation (dropout is disabled)
            out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            
            # Calculate validation loss
            val_loss = criterion(out[graph_data.val_mask], graph_data.y[graph_data.val_mask])
            val_losses.append(val_loss.item())
            
            # Calculate F1 scores for both train and val sets to check for overfitting
            train_preds = (torch.sigmoid(out[graph_data.train_mask]) > 0.5).float()
            val_preds = (torch.sigmoid(out[graph_data.val_mask]) > 0.5).float()
            
            train_f1 = f1_score(graph_data.y[graph_data.train_mask].cpu(), train_preds.cpu(), average="micro")
            val_f1 = f1_score(graph_data.y[graph_data.val_mask].cpu(), val_preds.cpu(), average="micro")
            
            train_f1_scores.append(train_f1)
            val_f1_scores.append(val_f1)

        # --- Live Plotting ---
        ax1.clear()
        ax2.clear()
        
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='orange')
        ax1.set_title('Loss over Epochs'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
        
        ax2.plot(train_f1_scores, label='Training F1 Score', color='blue')
        ax2.plot(val_f1_scores, label='Validation F1 Score', color='orange')
        ax2.set_title('Micro F1 Score over Epochs'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1 Score'); ax2.legend(); ax2.grid(True)
        
        fig.tight_layout()
        clear_output(wait=True)
        display(fig)
        time.sleep(0.1)

    plt.close(fig)
    print(f"\nTraining Complete. Final Validation F1 Score: {val_f1_scores[-1]:.4f}")


'''
# --- Main execution block to run the full process ---
if __name__ == '__main__':
    # This block demonstrates the complete workflow from start to finish.
    
    # --- A. Create Dummy AnnData object (replace with your sc.read_h5ad) ---
    print("Creating dummy AnnData object for demonstration...")
    n_obs, n_vars, n_pcs = 1000, 500, 50
    n_classes = 10
    adata = sc.AnnData(sparse_random(n_obs, n_vars, density=0.1, format='csr'))
    adata.obsm['X_pca'] = np.random.rand(n_obs, n_pcs)
    adata.obsm['X_pert_pos'] = np.random.rand(n_obs, 24)
    adata.obsp['connectivities'] = sparse_random(n_obs, n_obs, density=0.01, format='csr')
    adata.obsp['distances'] = sparse_random(n_obs, n_obs, density=0.01, format='csr')
    # Create dummy perturbation labels
    pert_names = [f'g{i}' for i in range(n_classes)] + [f'g{i}+g{j}' for i in range(3) for j in range(3,6)] + ['control']
    adata.obs['perturbation_name'] = np.random.choice(pert_names, n_obs)
    
    # --- B. Preprocess Labels (as you would in your notebook) ---
    parsed_labels = [p.split('+') if p != 'control' else [] for p in adata.obs['perturbation_name']]
    mlb = MultiLabelBinarizer()
    # Fit on all possible labels to ensure consistency
    all_possible_genes = list(set([item for sublist in parsed_labels for item in sublist]))
    mlb.fit([all_possible_genes])
    
    # --- C. Run the full pipeline ---
    # 1. Prepare all the data into the graph format
    graph_data = prepare_graph_data(adata, mlb, parsed_labels)

    # 2. Instantiate the model with the correct dimensions from the graph object
    model = EdgeConditionedGAT(
        in_channels=graph_data.num_node_features,
        edge_dim=graph_data.num_edge_features,
        hidden_channels=32,
        out_channels=len(mlb.classes_), # Use length of fitted classes
        heads=8
    )

    # 3. Set up optimizer and loss, then run training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    run_training(graph_data, model, optimizer, criterion)

'''