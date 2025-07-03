import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import wandb

from dataset import HD_Dataset, CP_Dataset
from graph import build_HC, build_DP
from models import iCAM
from train import MultiTaskTrainer

def main():
    wandb.init(project="iCAM-Net", name="Run_TCM_suite")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    lr = 1e-4
    weight_decay = 1e-5
    num_epochs = 50
    alpha_cp = 5e-2
    batch_size = 128
    hidden_dim = 128
    out_dim = 64
    n_layers = 3
    model_name = "iCAM-Net"
    wandb.config.update({
        "lr": lr,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "alpha_cp": alpha_cp,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "n_layers": n_layers,
        "model_name": model_name
    })

    hc_path = "../data/H_C_TCM.csv"
    ce_path = "../data/ce_TCM.txt"
    dp_path = "../data/D_P_TCM.csv"
    pe_path = "../data/pe_TCM.txt"

    H_C, X_C, herb2idx, comp2idx, herb2comp = build_HC(hc_path, ce_path, device)
    D_P, X_P, disease2idx, prot2idx, disease2prot = build_DP(dp_path, pe_path, device)

    print("[Step] Hypergraph and features loaded.")

    hd_pos = "../data/H_D_TCM.csv"
    hd_neg = "../data/H_D_TCM_neg.csv"
    hd_full_dataset = HD_Dataset(hd_pos, hd_neg, herb2idx, disease2idx, seed=42)
    full_size = len(hd_full_dataset)
    train_size = int(0.8 * full_size)
    val_size = int(0.1 * full_size)
    test_size = full_size - train_size - val_size

    hd_train_dataset, hd_val_dataset, hd_test_dataset = random_split(
        hd_full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
        )
    
    hd_train_loader = DataLoader(hd_train_dataset, batch_size=batch_size, shuffle=True)
    hd_val_loader = DataLoader(hd_val_dataset, batch_size=batch_size, shuffle=False)
    hd_test_loader = DataLoader(hd_test_dataset, batch_size=batch_size, shuffle=False)
    
    cp_pos = "../data/C_P_TCM.csv"
    cp_neg = "../data/C_P_TCM_neg.csv"
    cp_full_dataset = CP_Dataset(cp_pos, cp_neg, comp2idx, prot2idx, seed=42)
    cp_full_size = len(cp_full_dataset)
    cp_train_size = int(0.8 * cp_full_size)
    cp_val_size = int(0.1 * cp_full_size)
    cp_test_size = cp_full_size - cp_train_size - cp_val_size

    cp_train_dataset, cp_val_dataset, cp_test_dataset = random_split(
        cp_full_dataset, [cp_train_size, cp_val_size, cp_test_size], 
        generator=torch.Generator().manual_seed(42)
        )
    
    cp_train_loader = DataLoader(cp_train_dataset, batch_size=batch_size, shuffle=True)
    cp_val_loader = DataLoader(cp_val_dataset, batch_size=batch_size, shuffle=False)
    cp_test_loader = DataLoader(cp_test_dataset, batch_size=batch_size, shuffle=False)

    print(f"[Info]HD task => train{len(hd_train_dataset)}, val{len(hd_val_dataset)}, test{len(hd_test_dataset)}")
    print(f"[Info]CP task => train{len(cp_train_dataset)}, val{len(cp_val_dataset)}, test{len(cp_test_dataset)}")

    in_dim = X_C.shape[1]

    model = iCAM(
        herb_graph=H_C,
        disease_graph=D_P,
        X_C=X_C,
        X_P=X_P,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_layers=n_layers,
        herb2comp=herb2comp,
        disease2prot=disease2prot,
        return_attn=True,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    trainer = MultiTaskTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        model_name=model_name,
        alpha_cp=alpha_cp,
        hd_train_loader=hd_train_loader,
        hd_val_loader=hd_val_loader,
        hd_test_loader=hd_test_loader,
        cp_train_loader=cp_train_loader,
        cp_val_loader=cp_val_loader,
        cp_test_loader=cp_test_loader,
        use_wandb=True,
        result_root="./result"
    )

    trainer.train()

if __name__ == "__main__":
    main()
