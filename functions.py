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
import pyensembl
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
import modelCROSSATTN


def create_positional_embedding(adata, ):
    data = pyensembl.EnsemblRelease(109)
    data.download()
    data.index()

    chromosome_lengths = {
    #from https://www.ncbi.nlm.nih.gov/grc/human/data
    # Autosomes
    '1': 231223641,
    '2': 240863511,
    '3': 198255541,
    '4': 189962376,
    '5': 181358067,
    '6': 170078524,
    '7': 158970135,
    '8': 144768136,
    '9': 122084564,
    '10': 133263006,
    '11': 134634058,
    '12': 133137821,
    '13': 97983128,
    '14': 91660769,
    '15': 85089576,
    '16': 83378703,
    '17': 83481871,
    '18': 80089650,
    '19': 58440758,
    '20': 63944268,
    '21': 40088623,
    '22': 40181019,
    # Sex Chromosomes
    'X': 154893034,
    'Y': 26452288
    }

    chr_names = [str(i) for i in range(1, 23)] + ['X', 'Y']
    chromosome_map = {name: i for i, name in enumerate(chr_names)}

    gene_to_pos_encoding = {}
    zero_encoding = np.zeros(24)

    for gene_name in tqdm(adata.var_names):
        try:
            gene = data.genes_by_name(gene_name)
            encoding = np.zeros(24)
            contig = gene[0].contig
            if pd.notna(contig) and contig in chromosome_map:
                chr_index = chromosome_map[contig]
                chr_length = chromosome_lengths.get(contig, 0)
                if chr_length > 0:
                    encoding[chr_index] = gene[0].start / chr_length
            gene_to_pos_encoding[gene_name] = encoding
        except (ValueError, KeyError):
            continue 



    final_positional_encoding = []

    for perturbation_string in tqdm(adata.obs['perturbation_name']):
        if perturbation_string == 'control':
            final_positional_encoding.append(zero_encoding)
            continue

        gene_names_in_pert = perturbation_string.split('+')
        
        encodings_for_this_cell = []
        for gene_name in gene_names_in_pert:
            encoding = gene_to_pos_encoding.get(gene_name, zero_encoding)
            encodings_for_this_cell.append(encoding)

        averaged_encoding = np.mean(encodings_for_this_cell, axis=0)
        final_positional_encoding.append(averaged_encoding)

    final_positional_encoding = np.array(final_positional_encoding)

    print(f"\nSUCCESS: Final matrix generated with shape {final_positional_encoding.shape}")
    return final_positional_encoding

def create_masks(adata):
    control_mask = adata.obs['perturbation_name'] == "control"
    double_mask = adata.obs['perturbation_name'].str.contains("+",regex=False)
    single_mask = ~(control_mask | double_mask)
    masks = [control_mask,single_mask,double_mask]

    return masks

def split_data(adata, final_positional_encoding, masks, labels_int):
    expression_data = adata.X
    positional_data = final_positional_encoding
        
        # --- Create the full dataset object ---
    full_dataset = modelCROSSATTN.PerturbationMultiModalDataset(expression_data, positional_data, labels_int)



    #-------------basic splitting---------------
    '''    # --- Split indices for training and validation ---
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.3, 
        random_state=102
    )
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)'''

    #—————————————ADVANCED SPLITTING————————————————#
    # get perturbation labels for each cell
    pert_names = np.array(adata.obs['perturbation_name'])

    # unique condition names
    single_conditions = np.unique(pert_names[masks[1]])
    double_conditions = np.unique(pert_names[masks[2]])

    # --- Split perturbation *conditions* ---
    train_doubles, test_doubles = train_test_split(
        double_conditions,
        test_size=0.3,          # 30% of doubles unseen for testing
        random_state=102
    )

    # training conditions include all singles, controls, and 70% of doubles
    train_conditions = np.concatenate([single_conditions, ['control'], train_doubles])
    test_conditions  = test_doubles   # test on unseen doubles only

    # --- Map condition names to cell indices ---
    train_indices = np.where(np.isin(pert_names, train_conditions))[0]
    test_indices  = np.where(np.isin(pert_names, test_conditions))[0]

    # sanity checks
    print(f"Train cells: {len(train_indices)}  |  Test cells: {len(test_indices)}")
    print(f"Train perturbations: {len(train_conditions)}  |  Test perturbations: {len(test_conditions)}")
    print("Overlap between train/test perturbations:", np.intersect1d(train_conditions, test_conditions))

    # --- Make datasets ---
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset  = Subset(full_dataset, test_indices)

    return train_dataset, test_dataset

def plot(fig, ax1, ax2, ax3, train_losses, val_losses, train_f1_scores, val_f1_scores, acc1= None, acc5=None):
    # --- 2. Live Plotting Logic ---
    # Clear the previous plot from each axis
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    # Plot Loss Curves
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='orange')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot F1 Score Curves
    ax2.plot(train_f1_scores, label='Training F1 Score', color='blue')
    ax2.plot(val_f1_scores, label='Validation F1 Score', color='orange')
    ax2.set_title('Micro F1 Score over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True)

    '''    ax3.plot(acc1, label='acc1 score', color='blue')
    ax3.plot(acc5, label='acc5 score', color='orange')
    ax3.set_title('Topk Score over Epochs')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Percentage')
    ax3.legend()
    ax3.grid(True)
    '''

    fig.tight_layout()
    
    # Redraw the plot in the same cell
    clear_output(wait=True)
    display(fig)
    time.sleep(0.1) # A small pause to ensure the plot renders smoothly
    

def topk_accuracy(outputs, labels, k=5):
    """
    outputs: torch.Tensor [N, C] (logits or probabilities)
    labels:  torch.Tensor [N] (integer class IDs)
    k: int, how many top predictions to consider

    Returns: float, top-k accuracy in [0,1]
    """
    # Get the indices of the top-k predictions for each sample
    topk = outputs.topk(k, dim=1).indices  # shape [N, k]

    # Compare: does the true label appear in any of the top-k slots?
    correct = topk.eq(labels.view(-1, 1))  # shape [N, k] of True/False

    # If True anywhere in a row, it's correct
    correct_any = correct.any(dim=1)

    # Mean accuracy
    return correct_any.float().mean().item()