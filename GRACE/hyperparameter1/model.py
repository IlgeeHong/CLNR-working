#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim))
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNConv(hid_dim, hid_dim))
            self.convs.append(GCNConv(hid_dim, out_dim))
    def forward(self, x, edge_index):
        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](x, edge_index))
        x = self.convs[-1](x, edge_index)
        return x

class GRACE(nn.Module):
    def __init__(self, in_dim, hid_dim, proj_hid_dim, n_layers, tau = 0.5):
        super().__init__()
        self.gcn = GCN(in_dim, hid_dim, hid_dim, n_layers)
        self.fc1 = nn.Linear(hid_dim, proj_hid_dim)
        self.fc2 = nn.Linear(proj_hid_dim, hid_dim)
        self.tau = tau
        
    def get_embedding(self, data):
        out = self.gcn(data.x, data.edge_index)
        return out.detach()

    def forward(self, data1, data2):
        z1 = self.gcn(data1.x, data1.edge_index)
        z2 = self.gcn(data2.x, data2.edge_index)
        return z1, z2
    
    def projection(self, z):
        z = F.elu(self.fc1(z))
        h = self.fc2(z)
        return h
    
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1, z2, mean = True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret