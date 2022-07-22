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
            x = F.relu(self.convs[i](x, edge_index)) # nn.PReLU
        x = self.convs[-1](x, edge_index)
        return x


class Discriminator(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.layer = nn.Bilinear(out_dim, out_dim, 1)

    def forward(self, c1, c2, h1, h2, h3, h4):
        c_x1 = c1.expand_as(h1).contiguous()     
        c_x2 = c2.expand_as(h2).contiguous()     
        sc1 = self.layer(h2, c_x1) # .t() # data
        sc2 = self.layer(h1, c_x2) # .t() 
        sc3 = self.layer(h4, c_x1) # .t() # diffusion
        sc4 = self.layer(h3, c_x2) # .t() 
        logits = torch.cat((sc1, sc2, sc3, sc4)).flatten() #, 1
        # pdb.set_trace()
        return logits

class Readout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h):
        return torch.mean(h, 1, keepdim=True)

class MVGRL(nn.Module):
    def __init__(self, in_dim, out_dim): #  hid_dim
        super().__init__()
        self.encoder = GCNConv(in_dim, out_dim)
        self.diffusion_encoder = GCNConv(in_dim, out_dim, normalize = False)
        self.act = nn.PReLU()
        self.read = Readout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim)
    
    def get_embedding(self, data, diff):
        h1 = self.act(self.encoder(data.x, data.edge_index))
        h2 = self.act(self.diffusion_encoder(data.x, diff.edge_index, diff.edge_weight))
        return (h1 + h2).detach()

    def forward(self, data, diff, shuf_x):
        h1 = self.act(self.encoder(data.x, data.edge_index))
        h2 = self.act(self.diffusion_encoder(data.x, diff.edge_index, diff.edge_weight))
        h3 = self.act(self.encoder(shuf_x, data.edge_index))
        h4 = self.act(self.diffusion_encoder(shuf_x, diff.edge_index, diff.edge_weight))

        c1 = self.sigm(self.read(h1))
        c2 = self.sigm(self.read(h2))
        
        out = self.disc(c1, c2, h1, h2, h3, h4)

        return out