import torch
import torch.nn.functional as F
from torch import nn
import dgl.sparse as dglsp

class HGNN(nn.Module):
    """
    Hyper Graph Neural Network(HGNN) with skip connection, implemented by dgl
    """
    def __init__(self, H, in_dim, out_dim, hidden_dim, n_layers, device="cuda:1"):
        super(HGNN, self).__init__()
        self.device = device
        H = H.to(device)
        self.H = H      # shape: (num_nodes, num_edges)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #degree of nodes and edges
        d_V = H.sum(1)
        d_E = H.sum(0)
        d_V_invsqrt = d_V.pow(-0.5)
        d_E_inv = d_E.pow(-1)
        
        Dv_dgl = dglsp.diag(d_V_invsqrt).to(device)
        De_dgl = dglsp.diag(d_E_inv).to(device)
        W = dglsp.identity((d_E.shape[0], d_E.shape[0])).to(device)

        aggregator = Dv_dgl @ H @ W @De_dgl     # shape:(num_nodes, num_edges)
        self.node2edge = aggregator.T
        self.edge2node = aggregator

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                in_size = in_dim
            else:
                in_size = hidden_dim
            
            if i <n_layers - 1:
                out_size = hidden_dim
            else:
                out_size = out_dim
            
            layer = nn.ModuleDict({
                'fc1' : nn.Linear(in_size, out_size),
                'proj1' : nn.Linear(in_size, out_size),
                'fc2' : nn.Linear(out_size, out_size),
                'proj2' : nn.Linear(out_size, out_size)
            })

            self.layers.append(layer)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, X):
        for i, layer in enumerate(self.layers):
            X_skip1 = layer['proj1'](X)
            X_drop1 = self.dropout(X)
            X_fc1 = layer['fc1'](X_drop1)

            E = self.node2edge @ X_fc1
            E = F.leaky_relu(E, negative_slope=0.1)
            X_new = self.edge2node @ E
            X = F.leaky_relu(X_new + X_skip1, negative_slope=0.1)

            X_skip2 = layer['proj2'](X)
            X_drop2 = self.dropout(X)
            X_fc2 = layer['fc2'](X_drop2)

            E2 = self.node2edge @ X_fc2
            E2 = F.leaky_relu(E2, negative_slope=0.1)
            X_new2 = self.edge2node @ E2
            X = F.leaky_relu(X_new2 + X_skip2, negative_slope=0.1)
        
        E_final = self.node2edge @ X

        return E_final, X
