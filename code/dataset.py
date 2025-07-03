import torch
import pandas as pd
import random
from torch.utils.data import Dataset

class HD_Dataset(Dataset):
    def __init__(self, pos, neg, herb2idx, disease2idx, seed=42):
        super().__init__()
        random.seed(seed)

        df_pos = pd.read_csv(pos)
        df_neg = pd.read_csv(neg)
        df_pos['label'] = 1
        df_neg['label'] = 0

        pos_count = len(df_pos)
        neg_count = len(df_neg)
        sample_size = min(pos_count, neg_count)

        df_neg_sampled = df_neg.sample(n=sample_size, random_state=seed)

        df_all = pd.concat([df_pos, df_neg_sampled], ignore_index=True)
        df_all = df_all.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        def map_hid(h_id):
            return herb2idx[h_id] if h_id in herb2idx else -1
        
        def map_did(d_id):
            return disease2idx[d_id] if d_id in disease2idx else -1
        
        df_all['herbIndex'] = df_all['herbid'].apply(map_hid)
        df_all['diseaseIndex'] = df_all['diseaseId'].apply(map_did)
        df_all = df_all[(df_all['herbIndex'] >= 0) & (df_all['diseaseIndex'] >= 0)].reset_index(drop=True)

        self.df = df_all

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        herb_idx = row['herbIndex']
        disease_idx = row['diseaseIndex']
        label = row['label']

        return (torch.tensor([herb_idx], dtype=torch.long),
                torch.tensor([disease_idx], dtype=torch.long),
                torch.tensor([label], dtype=torch.float32))

class CP_Dataset(Dataset):
    def __init__(self, pos, neg, comp2idx, protein2idx, seed=42):
        super().__init__()
        random.seed(seed)

        df_pos = pd.read_csv(pos)

        df_neg = pd.read_csv(neg)
        df_pos['label'] = 1
        df_neg['label'] = 0
        pos_count = len(df_pos)
        neg_count = len(df_neg)
        sample_size = min(pos_count, neg_count)
        df_neg_sampled = df_neg.sample(n=sample_size, random_state=seed)
        df_all = pd.concat([df_pos, df_neg_sampled], ignore_index=True)
        df_all = df_all.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        def map_cid(cid):
            return comp2idx[cid] if cid in comp2idx else -1

        def map_pid(gid):
            return protein2idx[gid] if gid in protein2idx else -1

        df_all['compIndex'] = df_all['cid'].apply(map_cid)
        df_all['proteinIndex'] = df_all['geneId'].apply(map_pid)
        df_all = df_all[(df_all['compIndex'] >= 0) & (df_all['proteinIndex'] >= 0)].reset_index(drop=True)

        self.df = df_all


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        comp_idx = row['compIndex']
        prot_idx = row['proteinIndex']
        label_cp = row['label']

        return (torch.tensor([comp_idx], dtype=torch.long),
                torch.tensor([prot_idx], dtype=torch.long),
                torch.tensor([label_cp], dtype=torch.float32))