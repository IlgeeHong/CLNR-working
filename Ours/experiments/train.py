#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pdb
from torch_geometric.nn import GCNConv

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)
        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, x, _):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        return x

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

class CLGR(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, tau, use_mlp=False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.tau = tau

    def get_embedding(self, data):
        out = self.backbone(data.x, data.edge_index)
        return out.detach()

    def forward(self, data1, data2):
        h1 = self.backbone(data1.x, data1.edge_index)
        h2 = self.backbone(data2.x, data2.edge_index)
        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)
        pdb.set_trace()
        return z1, z2
    
    def sim(self, z1, z2, indices, refl=None):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        f = lambda x: torch.exp(x / self.tau) 
        if indices is not None:
            z2_new = z2[indices,:]
            sim = f(torch.mm(z1, z2_new.t()))
            diag = f(torch.mm(z1,z2.t()).diag())
        else:
            sim = f(torch.mm(z1, z2.t()))
            diag = f(torch.mm(z1, z2.t()).diag())
        return sim, diag
    
    def semi_loss(self, z1, z2, indices):
        refl_sim, refl_diag = self.sim(z1, z1, indices)
        between_sim, between_diag = self.sim(z1, z2, indices)
        if indices is not None:
            refl_diag_temp = refl_diag.clone()
            refl_diag_temp[~indices] = 0.0
            refl_diag_neg = refl_diag_temp.clone()
        else:
            refl_diag_temp = refl_diag.clone()
            refl_diag_neg = refl_diag_temp.clone()
        semi_loss = - torch.log(between_diag / (between_sim.sum(1) + refl_sim.sum(1) - refl_diag_neg))
        return semi_loss

    def loss(self, z1, z2, k=None, mean=True):
        if k is not None:
            N = z1.shape[0]
            indices = torch.LongTensor(random.sample(range(N), k))
        else:
            indices = None
        l1 = self.semi_loss(z1, z2, indices)
        l2 = self.semi_loss(z2, z1, indices)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

class SupCLGR(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, tau = 0.5, use_mlp = False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.tau = tau

    def get_embedding(self, data):
        out = self.backbone(data.x, data.edge_index)
        return out.detach()

    def forward(self, data1, data2):
        h1 = self.backbone(data1.x, data1.edge_index)
        h2 = self.backbone(data2.x, data2.edge_index)
        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)
        return z1, z2
    
    def sim(self, z1, z2, pos_idx):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        f = lambda x: torch.exp(x / self.tau) 
        # if indices is not None:
        #     z2_new = z2[indices,:]
        #     sim = f(torch.mm(z1, z2_new.t()))
        #     diag = f(torch.mm(z1,z2.t()).diag())
        #     sim_pos_temp1 = f(torch.mm(z1, z2.t()))
        #     sim_pos_temp2 = sim_pos_temp1.clone()
        #     sim_pos_temp2[~pos_idx] = 0
        #     sim_pos = sim_pos_temp2.clone()
        #     sim_pos_sum = sim_pos.sum(1)
        # else:
        sim = f(torch.mm(z1, z2.t()))
        diag = f(torch.mm(z1, z2.t()).diag())
        sim_pos_temp1 = sim.clone()
        sim_pos_temp1[~pos_idx] = 0
        sim_pos = sim_pos_temp1.clone()
        sim_pos_sum = sim_pos.sum(1)
        return sim, diag, sim_pos_sum

    def semi_loss(self, data, z1, z2, num_class, train_idx, indices):
        class_idx = []
        for c in range(num_class):
            index = (data.y == c) * train_idx
            class_idx.append(index)
        class_idx = torch.stack(class_idx).bool()
        pos_idx = class_idx[data.y]
        pos_idx[~train_idx] = False
        pos_idx.fill_diagonal_(True)
        
        refl_sim, refl_diag, refl_pos_sum = self.sim(z1, z1, pos_idx, indices)
        between_sim, _, between_pos_sum = self.sim(z1, z2, pos_idx, indices)
        num_per_class = pos_idx.sum(1)

        # if indices is not None:
        #     refl_diag_temp = refl_diag.clone()
        #     refl_diag_temp[~indices] = 0.0
        #     refl_diag_neg = refl_diag_temp.clone()
        # else:
        refl_diag_temp = refl_diag.clone()
        refl_diag_neg = refl_diag_temp.clone()
        
        semi_loss = -torch.log(
            (1/(2*num_per_class-1))*(between_pos_sum + refl_pos_sum - refl_diag) / (between_sim.sum(1) + refl_sim.sum(1) - refl_diag_neg)
            )
        return semi_loss
        
    def loss(self, data, z1, z2, num_class, train_idx, k=None, mean=True):
        if k is not None:
            N = z1.shape[0]
            indices = torch.LongTensor(random.sample(range(N), k))   
        else:
            indices = None
        l1 = self.semi_loss(data, z1, z2, num_class, train_idx, indices)
        l2 = self.semi_loss(data, z2, z1, num_class, train_idx, indices)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
