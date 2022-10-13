import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pdb
import copy
import numpy as np
from dbn import *
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
        return z1, z2
    
    def sim(self, z1, z2, indices):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        f = lambda x: torch.exp(x / self.tau) 
        if indices is not None:
            z1_new = z1[indices,:]
            z2_new = z2[indices,:]
            sim = f(torch.mm(z1_new, z2_new.t()))
            diag = sim.diag()
        else:
            sim = f(torch.mm(z1, z2.t()))
            diag = f(torch.mm(z1, z2.t()).diag())
        return sim, diag
    
    def semi_loss(self, z1, z2, indices):
        refl_sim, refl_diag = self.sim(z1, z1, indices)
        between_sim, between_diag = self.sim(z1, z2, indices)
        semi_loss = - torch.log(between_diag / (between_sim.sum(1) + refl_sim.sum(1) - refl_diag))
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

class GRACE(nn.Module):
    def __init__(self, in_dim, hid_dim, proj_hid_dim, n_layers, tau = 0.5, use_mlp = False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, hid_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, hid_dim)

        self.fc1 = nn.Linear(hid_dim, proj_hid_dim)
        self.fc2 = nn.Linear(proj_hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, hid_dim)
        self.tau = tau
        
    def get_embedding(self, data):
        out = self.backbone(data.x, data.edge_index)
        return out.detach()

    def forward(self, data1, data2):
        z1 = self.backbone(data1.x, data1.edge_index)
        z2 = self.backbone(data2.x, data2.edge_index)
        return z1, z2
    
    def projection(self, z, layer="nonlinear-hid"):
        if layer == "nonlinear-hid":
            z = F.elu(self.fc1(z))
            h = self.fc2(z)
        elif layer == "nonlinear":
            h = F.elu(self.fc3(z))
        elif layer == "linear":
            h = self.fc3(z)
        elif layer == "standard":
            h = (z - z.mean(0)) / z.std(0)
        elif layer == 'dbn':
            dbn = DBN(device=z.device, num_features=z.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
            h = dbn(z)          
        return h
    
    def sim(self, z1, z2, indices):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        f = lambda x: torch.exp(x / self.tau) 
        if indices is not None:
            z1_new = z1[indices,:]
            z2_new = z2[indices,:]
            sim = f(torch.mm(z1_new, z2_new.t()))
            diag = sim.diag()
        else:
            sim = f(torch.mm(z1, z2.t()))
            diag = f(torch.mm(z1, z2.t()).diag())
        return sim, diag

    def semi_loss(self, z1, z2, indices):
        refl_sim, refl_diag = self.sim(z1, z1, indices)
        between_sim, between_diag = self.sim(z1, z2, indices)
        semi_loss = - torch.log(between_diag / (between_sim.sum(1) + refl_sim.sum(1) - refl_diag))
        return semi_loss

    def loss(self, z1, z2, layer="nonlinear-hid", k=None, mean = True):
        if k is not None:
            N = z1.shape[0]
            indices = torch.LongTensor(random.sample(range(N), k))
        else:
            indices = None
        h1 = self.projection(z1, layer)
        h2 = self.projection(z2, layer)
        l1 = self.semi_loss(h1, h2, indices)
        l2 = self.semi_loss(h2, h1, indices)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

class Encoder(nn.Module):
    def __init__(self, layer_config, dropout=None, project=False):
        super().__init__()
        self.conv1 = GCNConv(layer_config[0], layer_config[1])
        # self.bn1 = nn.BatchNorm1d(layer_config[1], momentum = 0.01)
        self.prelu1 = nn.PReLU()
        self.conv2 = GCNConv(layer_config[1],layer_config[2])
        # self.bn2 = nn.BatchNorm1d(layer_config[2], momentum = 0.01)
        self.prelu2 = nn.PReLU()
        self.conv3 = GCNConv(layer_config[2],layer_config[3])
        self.prelu3 = nn.PReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.prelu1(x) # self.bn1(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.prelu2(x) # self.bn2(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.prelu3(x) # self.bn2(x)
        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

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
    
def lossfunc(x, y):
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
    def __init__(self, layer_config, pred_hid, epochs, dropout=0.0, moving_average_decay=0.99):
        super().__init__()
        self.student_encoder = Encoder(layer_config=layer_config, dropout=dropout)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        rep_dim = layer_config[-1]
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.Linear(pred_hid, rep_dim))
        self.student_predictor.apply(init_weights) 
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None
    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)
    def get_embedding(self, data):
        out = self.student_encoder(data.x, data.edge_index)
        return out.detach()
    def forward(self, data1, data2):
        x1 = data1.x
        edge_index_v1 = data1.edge_index
        edge_weight_v1 = None
        x2 = data2.x
        edge_index_v2 = data2.edge_index
        edge_weight_v2 = None
        v1_student = self.student_encoder(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
        v2_student = self.student_encoder(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        
        with torch.no_grad():
            v1_teacher = self.teacher_encoder(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
            v2_teacher = self.teacher_encoder(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)
            
        loss1 = lossfunc(v1_pred, v2_teacher.detach())
        loss2 = lossfunc(v2_pred, v1_teacher.detach())

        loss = loss1 + loss2
        return v1_student, v2_student, loss.mean()


