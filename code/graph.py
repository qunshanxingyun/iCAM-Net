import dgl.sparse as dglsp
import torch
import pandas as pd

"""
Functions for hyper graph building
"""

def _build_herb2comp(H_C):
    coo = H_C.coalesce()
    comp_indices, herb_indcies = coo.indices()
    herb2comp = {}
    comp_list = comp_indices.tolist()
    herb_list = herb_indcies.tolist()

    for comp_idx, herb_idx in zip(comp_list, herb_list):
        if herb_idx not in herb2comp:
            herb2comp[herb_idx] = []
        herb2comp[herb_idx].append(comp_idx)
    return herb2comp

def _build_disease2prot(D_P):
    coo = D_P.coalesce()
    prot_indices, disease_indices = coo.indices()
    disease2prot = {}
    prot_list = prot_indices.tolist()
    disease_list = disease_indices.tolist()

    for prot_idx, disease_idx in zip(prot_list, disease_list):
        if disease_idx not in disease2prot:
            disease2prot[disease_idx] = []
        disease2prot[disease_idx].append(prot_idx)
    return disease2prot

def comp_embedding(ce_path):
    comp_embedding_map = {}
    with open(ce_path, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            comp_cid = parts[0]
            comp_emb = parts[1].split()
            emb_val = []
            for x in comp_emb:
                emb_val.append(float(x))
            comp_embedding_map[comp_cid] = emb_val
    return comp_embedding_map

def prot_embedding(pe_path):
    prot_embedding_map = {}
    with open(pe_path, 'r', encoding="UTF-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            prot_pid = parts[0]
            prot_emb = parts[1].split()
            emb_val = []
            for x in prot_emb:
                emb_val.append(float(x))
            prot_embedding_map[prot_pid] = emb_val
    return prot_embedding_map

def build_HC(hc_path, ce_path, device="cuda:1"):
    hc = pd.read_csv(hc_path)
    herb_ids = hc['herbid'].unique()
    comp_ids = hc['cid'].unique()
    herb2idx = {}
    for idx, id in enumerate(herb_ids):
        herb2idx[id] = idx
    comp2idx = {}
    for idx, id in enumerate(comp_ids):
        comp2idx[id] = idx
    herb_indices = []
    for herb_id in hc['herbid']:
        herb_indices.append(herb2idx[herb_id])
    comp_indices = []
    for comp_id in hc['cid']:
        comp_indices.append(comp2idx[comp_id])
    
    H_C = dglsp.spmatrix(torch.LongTensor([comp_indices, herb_indices])).to(device)
    comp_embedding_map = comp_embedding(ce_path=ce_path)
    n_comp = len(comp_ids)
    embed_dim = len(next(iter(comp_embedding_map.values())))
    X_comp = torch.zeros((n_comp, embed_dim), dtype=torch.float32).to(device)
    for comp_id, idx in comp2idx.items():
        cid_str = str(comp_id)
        if cid_str in comp_embedding_map:
            X_comp[idx] = torch.tensor(comp_embedding_map[cid_str], dtype=torch.float32)
        else:
            print(f"Warning: {cid_str} not found in compound embedding file")
    herb2comp = _build_herb2comp(H_C)

    return H_C, X_comp, herb2idx, comp2idx, herb2comp

def build_DP(dp_path, pe_path, device="cuda:1"):
    dp = pd.read_csv(dp_path)
    disease_ids = dp['diseaseId'].unique()
    prot_ids = dp['geneId'].unique()
    disease2idx = {}
    for idx, id in enumerate(disease_ids):
        disease2idx[id] = idx
    prot2idx = {}
    for idx, id in enumerate(prot_ids):
        prot2idx[id] = idx
    
    disease_indices = [disease2idx[did] for did in dp['diseaseId']]
    prot_indices = [prot2idx[pid] for pid in dp['geneId']]
    
    D_P = dglsp.spmatrix(torch.LongTensor([prot_indices, disease_indices])).to(device)
    prot_embedding_map = prot_embedding(pe_path)

    n_prot = len(prot_ids)
    
    embed_dim = len(next(iter(prot_embedding_map.values())))
    X_prot = torch.zeros((n_prot, embed_dim), dtype=torch.float32).to(device)
    for prot_id, idx in prot2idx.items():
        pid = str(prot_id)
        if pid in prot_embedding_map:
            X_prot[idx] = torch.tensor(prot_embedding_map[pid], dtype=torch.float32).to(device)
        else:
            print(f"Warning: {pid} not found in protein embedding file")
            
    disease2prot = _build_disease2prot(D_P)
    return D_P, X_prot, disease2idx, prot2idx, disease2prot