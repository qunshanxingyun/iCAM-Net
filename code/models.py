import torch
import torch.nn as nn
import torch.nn.functional as F 
from HGNN import HGNN

class CrossAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout = nn.Dropout(0.2)

        self.Wq_c = nn.Linear(d_in, d_out)
        self.Wk_c = nn.Linear(d_in, d_out)
        self.Wv_c = nn.Linear(d_in, d_out)

        self.Wq_p = nn.Linear(d_in, d_out)
        self.Wk_p = nn.Linear(d_in, d_out)
        self.Wv_p = nn.Linear(d_in, d_out)

        self.scale = d_out ** 0.5
    
    def forward_with_projection(self, Qc, Kc, Vc, Qp, Kp, Vp):
        raw_cp = torch.mm(Qc, Kp.T)/(self.scale)
        cp_score = F.softmax(raw_cp, dim=-1)
        comp_fused = torch.mm(cp_score, Vp)

        raw_pc = torch.mm(Qp, Kc.T)/(self.scale)
        pc_score = F.softmax(raw_pc, dim=-1)
        prot_fused = torch.mm(pc_score, Vc)

        return comp_fused, prot_fused

    def forward(self, Qc, Kc, Vc, Qp, Kp, Vp):
        Qc = self.Wq_c(Qc)
        Kc = self.Wk_c(Kc)
        Vc = self.Wv_c(Vc)

        Qp = self.Wq_p(Qp)
        Kp = self.Wk_p(Kp)
        Vp = self.Wv_p(Vp)

        Qc = self.dropout(Qc)
        Kc = self.dropout(Kc)
        Vc = self.dropout(Vc)

        Qp = self.dropout(Qp)
        Kp = self.dropout(Kp)
        Vp = self.dropout(Vp)

        return self.forward_with_projection(Qc, Kc, Vc, Qp, Kp, Vp)
    
class iCAM(nn.Module):
    def __init__(self, herb_graph, disease_graph, X_C, X_P, in_dim,
                 hidden_dim, out_dim, device, n_layers,
                 herb2comp, disease2prot, return_attn=False) :
        super(iCAM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device
        self.return_attn = return_attn
        self.dropout = nn.Dropout(0.1)
        self.herb2comp = herb2comp
        self.disease2prot = disease2prot

        self.herb_graph = herb_graph.to(device)
        self.disease_graph = disease_graph.to(device)
        self.X_C = X_C.to(device)
        self.X_P = X_P.to(device)

        self.hgnn_h = HGNN(herb_graph, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                           n_layers=n_layers,device=device)
        self.hgnn_d = HGNN(disease_graph, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                           n_layers=n_layers,device=device)
        
        self.cross_attn = CrossAttention(d_in=out_dim, d_out=out_dim)

        self.gating_c = nn.Linear(out_dim, out_dim, bias=True)
        self.gating_p = nn.Linear(out_dim, out_dim, bias=True)
        

        self.output_layer = nn.Linear(out_dim, 1, bias=True)    # dot product or MLP
        self.cp_predictor = bilinearInteraction(dim=out_dim)

    def gating(self, x, layer):
        gate = torch.sigmoid(layer(x))
        return x * gate
    
    def pooling(self, vecs, attn_weight, dim=1, pooling_type="max"):
        """
        Attention driven pooling methods
        """
        if vecs.size(0) == 1:
            return vecs.unsqueeze(0)
        if pooling_type == "max":
            if dim == 1:
                importance, _ = attn_weight.max(dim=dim, keepdim=True)      # compound
            else:
                importance, _ = attn_weight.transpose(0,1).max(dim=1, keepdim=True)     # protein
        
        elif pooling_type == "mean":
            if dim == 1:
                importance = attn_weight.mean(dim=dim, keepdim=True)      # compound
            else:
                importance = attn_weight.mean(dim=dim, keepdim=True).transpose(0,1)     # protein

        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}, must be 'mean' or 'max'")
        
        weights = torch.softmax(importance, dim=0)

        pooled = (vecs * weights).sum(dim=0, keepdim=True)
        return pooled
    
    def forward(self, herb_ids, disease_ids):
        batch_size = herb_ids.size(0)

        _, compound_embedding = self.hgnn_h(self.X_C)
        _, protein_embedding = self.hgnn_d(self.X_P)

        comp_q = self.cross_attn.Wq_c(compound_embedding)
        comp_k = self.cross_attn.Wk_c(compound_embedding)
        comp_v = self.cross_attn.Wv_c(compound_embedding)

        prot_q = self.cross_attn.Wq_p(protein_embedding)
        prot_k = self.cross_attn.Wk_p(protein_embedding)
        prot_v = self.cross_attn.Wv_p(protein_embedding)

        """
        Calculate the maximum length of the component/protein subset of herbs/diseases in one batch
        """ 
        max_comp = max(len(self.herb2comp[h.item()]) for h in herb_ids)
        max_prot = max(len(self.disease2prot[d.item()]) for d in disease_ids)

        """
        Create index and mask tensors for subsets for batch calculation
        """
        batch_comp_indices = torch.zeros(batch_size, max_comp, dtype=torch.long, device=self.device)
        batch_prot_indices = torch.zeros(batch_size, max_prot, dtype=torch.long, device=self.device)

        comp_mask = torch.zeros(batch_size, max_comp, dtype=torch.bool, device=self.device)
        prot_mask = torch.zeros(batch_size, max_prot, dtype=torch.bool, device=self.device)

        for b in range(batch_size):
            h_comps = self.herb2comp[herb_ids[b].item()]
            d_prots = self.disease2prot[disease_ids[b].item()]

            batch_comp_indices[b, :len(h_comps)] = torch.tensor(h_comps, device=self.device)
            batch_prot_indices[b, :len(d_prots)] = torch.tensor(d_prots, device=self.device)

            comp_mask[b, :len(h_comps)] = True
            prot_mask[b, :len(d_prots)] = True

        batch_comp_q = comp_q[batch_comp_indices]
        batch_comp_k = comp_k[batch_comp_indices]
        batch_comp_v = comp_v[batch_comp_indices]

        batch_prot_q = prot_q[batch_prot_indices]
        batch_prot_k = prot_k[batch_prot_indices]
        batch_prot_v = prot_v[batch_prot_indices]

        scale = self.cross_attn.scale

        batch_raw_score_cp = torch.einsum('bnc,bmc->bnm', batch_comp_q, batch_prot_k) / scale

        comp_mask_expanded = comp_mask.unsqueeze(-1)    # [batch, max_comp, 1]
        prot_mask_expanded = prot_mask.unsqueeze(1)     # [batch, 1, max_prot]

        attn_mask = comp_mask_expanded & prot_mask_expanded     # [batch, max_comp, max_prot]

        batch_raw_score_cp = batch_raw_score_cp.masked_fill(~attn_mask, -1e9)

        batch_attn_cp = F.softmax(batch_raw_score_cp, dim=-1)
        batch_comp_fused = torch.einsum('bnm,bmd->bnd', batch_attn_cp, batch_prot_v)

        batch_raw_score_pc = torch.einsum('bmc,bnc->bmn', batch_prot_q, batch_comp_k) / scale

        batch_raw_score_pc = batch_raw_score_pc.masked_fill(~attn_mask.transpose(1,2), -1e9)

        batch_attn_pc = F.softmax(batch_raw_score_pc, dim=-1)
        batch_prot_fused = torch.einsum('bmn,bnd->bmd', batch_attn_pc, batch_comp_v)

        batch_comp_fused = self.gating(batch_comp_fused, self.gating_c)
        batch_prot_fused = self.gating(batch_prot_fused, self.gating_p)

        """
        Create vecs for pooling process
        """
        batch_herb_vec = torch.zeros(batch_size, 1, self.out_dim, device=self.device)
        batch_disease_vec = torch.zeros(batch_size, 1, self.out_dim, device=self.device)

        for b in range(batch_size):
            n_comps = comp_mask[b].sum().item()
            n_prots = prot_mask[b].sum().item()

            if n_comps > 0 and n_prots > 0:
                comp_fused_sub = batch_comp_fused[b, :n_comps]
                prot_fused_sub = batch_prot_fused[b, :n_prots]
                attn_sub = batch_attn_cp[b, :n_comps, :n_prots]

                if n_comps > 1:
                    herb_vec = self.pooling(
                        vecs=comp_fused_sub, 
                        attn_weight=attn_sub, 
                        dim=1, pooling_type='max')
                else:
                    herb_vec = comp_fused_sub.unsqueeze(0)

                if n_prots > 1:
                    disease_vec = self.pooling(
                        vecs=prot_fused_sub,
                        attn_weight=attn_sub.transpose(0,1),
                        dim=1, pooling_type='max'
                    )
                else:
                    disease_vec = prot_fused_sub.unsqueeze(0)
                
                batch_herb_vec[b] = herb_vec
                batch_disease_vec[b]= disease_vec
        
        batch_herb_vec = self.dropout(batch_herb_vec)
        batch_disease_vec = self.dropout(batch_disease_vec)

        batch_logits = torch.sum(batch_herb_vec*batch_disease_vec, dim=-1)
        batch_scores = torch.sigmoid(batch_logits)

        if self.return_attn:
            attn_list = []
            for b in range(batch_size):
                n_comps = comp_mask[b].sum().item()
                n_prots = prot_mask[b].sum().item()
                attn_list.append(batch_attn_pc[b, :n_comps, :n_prots])
            return batch_scores.squeeze(-1), attn_list
        else:
            return batch_scores.squeeze(-1)
        
    
    def forward_cp_score(self, comp_ids, prot_ids):
        comp_ids = comp_ids.squeeze(-1)
        prot_ids = prot_ids.squeeze(-1)

        _, compound_embedding = self.hgnn_h(self.X_C)
        _, protein_embedding = self.hgnn_d(self.X_P)
        
        # 1. 为当前批次选择基础嵌入
        batch_comp_emb_base = compound_embedding[comp_ids]
        batch_prot_emb_base = protein_embedding[prot_ids]

        # 2. 计算V向量用于交叉更新
        batch_comp_v = self.cross_attn.Wv_c(batch_comp_emb_base)
        batch_prot_v = self.cross_attn.Wv_p(batch_prot_emb_base)
        
        # 3. 对基础嵌入应用门控
        gated_batch_comp_emb = self.gating(batch_comp_emb_base, self.gating_c)
        gated_batch_prot_emb = self.gating(batch_prot_emb_base, self.gating_p)

        # 4. 根据原始逻辑进行交叉更新: final_emb = gated_emb + V_vector_from_other_entity
        c_final = gated_batch_comp_emb + batch_prot_v
        p_final = gated_batch_prot_emb + batch_comp_v

        # 5. 使用双线性交互预测分数
        scores_cp = self.cp_predictor(c_final, p_final)
        
        return scores_cp


class bilinearInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bilinear = nn.Bilinear(in1_features=dim, in2_features=dim, out_features=1)
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x1, x2):
        logits = self.bilinear(x1, x2).squeeze(-1)
        return torch.sigmoid(logits + self.bias)
