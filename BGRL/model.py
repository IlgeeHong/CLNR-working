#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, norm = True):
        super().__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim, normalize = norm))
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize = norm))
            self.convs.append(GCNConv(hid_dim, out_dim, normalize = norm))

    def forward(self, x, edge_index, edge_weight = None):
        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_weight)) # nn.PReLU
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new
    
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class BGRL(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, pred_hid, moving_average_decay=0.99, epochs=1000):
        super().__init__()
        self.student_encoder = GCN(in_dim, hid_dim, out_dim, n_layers)    
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        rep_dim = out_dim
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.Linear(pred_hid, rep_dim))
        # self.student_predictor.apply(init_weights)
    
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def get_embedding(self, data):
        z = self.teacher_encoder(data.x, data.edge_index)
        return z.detach()    

    def forward(self, data1, data2):
        v1_student = self.student_encoder(data1.x, data1.edge_index, edge_weight=data1.edge_weight)
        v2_student = self.student_encoder(data2.x, data2.edge_index, edge_weight=data2.edge_weight)

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        
        with torch.no_grad():
            v1_teacher = self.teacher_encoder(data1.x, data1.edge_index, edge_weight=data1.edge_weight)
            v2_teacher = self.teacher_encoder(data2.x, data2.edge_index, edge_weight=data2.edge_weight)
            
        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())

        loss = loss1 + loss2
        return v1_student, v2_student, loss.mean()
