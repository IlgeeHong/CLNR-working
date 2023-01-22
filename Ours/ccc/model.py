import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from torch_geometric.nn import GCNConv
from dbn import *
from aug import *
import pdb
from torch_geometric.loader import NeighborLoader
# from ogb.nodeproppred import Evaluator

# CUDA support
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

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
            x = F.relu(self.convs[i](x, edge_index)) #
        x = self.convs[-1](x, edge_index)
        return x

class Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, tau, lambd, device, model="CLNR", use_mlp=False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.tau = tau
        self.lambd = lambd
        self.model = model
        self.device = device
        self.fc1 = nn.Linear(out_dim, out_dim * 2)
        self.fc2 = nn.Linear(out_dim * 2, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def get_embedding(self, data):
        out = self.backbone(data.x, data.edge_index) 
        # No projection head here
        return out.detach()

    def forward(self, data1, data2):
        # Encode the graph
        u = self.backbone(data1.x, data1.edge_index)
        v = self.backbone(data2.x, data2.edge_index)
        return u, v
        
    def projection(self, u, v):
        if self.model == "GRACE":
            u = F.elu(self.fc1(u))
            v = F.elu(self.fc1(v))
            z1 = self.fc2(u)
            z2 = self.fc2(v)
        elif self.model in ["CCA-SSG","CLNR","CLNR-unif","CLNR-align"]:
            z1 = (u - u.mean(0)) / u.std(0)
            z2 = (v - v.mean(0)) / v.std(0)
        elif self.model in "dCCA-SSG":
            dbn1 = DBN(device=u.device, num_features=u.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
            dbn2 = DBN(device=v.device, num_features=v.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
            z1 = dbn1(u)
            z2 = dbn2(v)
        elif self.model == "dCLNR":
            dbn1 = DBN(device=u.device, num_features=u.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
            dbn2 = DBN(device=v.device, num_features=v.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
            z1 = dbn1(u)
            z2 = dbn2(v)
        elif self.model == "bCLNR":
            z1 = self.bn(u)
            z2 = self.bn(v)
        else:
            z1 = u              
            z2 = v
        return z1, z2
    
    def sim(self, z1, z2, indices):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1[indices,:], z2[indices,:].T)

    def semi_loss(self, z1, z2, indices, loss_type='ntxent'):
        f = lambda x: torch.exp(x / self.tau)
        if loss_type in ["ntxent", "ntxent_cca"]:
            refl_sim = f(self.sim(z1, z1, indices))
            between_sim = f(self.sim(z1, z2, indices))   
            loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        elif loss_type == "ntxent-align":
            refl_sim = f(self.sim(z1, z1, indices))
            between_sim = f(self.sim(z1, z2, indices))
            l1 = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)        
            l2 = (z1-z2).norm(dim=1).pow(2).mean()
            loss = l1 + (l2 * self.lambd)
        elif loss_type == "ntxent-uniform":
            refl_sim = f(self.sim(z1, z1, indices))
            between_sim = f(self.sim(z1, z2, indices))
            l1 = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
            z1 = F.normalize(z1)
            sq_pdist = torch.pdist(z1, p=2).pow(2)
            l2 = sq_pdist.mul(-2).exp().mean()
            loss = l1 + (l2 * self.lambd)
        return loss

    def loss(self, u, v, k, loss_type='ntxent', mean = True):
        if k is not None:
            N = u.shape[0]
            indices = torch.LongTensor(random.sample(range(N), k))
        else:
            indices = None
        z1, z2 = self.projection(u, v)    
        if loss_type == "ntxent":
            l1 = self.semi_loss(z1, z2, indices, loss_type)
            l2 = self.semi_loss(z2, z1, indices, loss_type)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        if loss_type == "ntxent_cca":
            l1 = self.semi_loss(z1, z2, indices, loss_type)
            l2 = self.semi_loss(z2, z1, indices, loss_type)
            inv = (l1 + l2) * 0.5
            inv = inv.mean() if mean else inv.sum()   
            c = torch.mm(z1.T, z2)
            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)
            c = c #/ N
            c1 = c1 #/ N
            c2 = c2 #/ N 
            iden = torch.tensor(np.eye(c.shape[0])).to(self.device)
            loss_dec1 = (iden - c1).pow(2).sum()
            loss_dec2 = (iden - c2).pow(2).sum()
            ret = inv + self.lambd * (loss_dec1 + loss_dec2)
        elif loss_type == 'cca':
            N = z1.shape[0]
            c = torch.mm(z1.T, z2)
            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)
            c = c / N
            c1 = c1 / N
            c2 = c2 / N
            loss_inv = - torch.diagonal(c).sum()
            iden = torch.tensor(np.eye(c.shape[0])).to(self.device)
            loss_dec1 = (iden - c1).pow(2).sum()
            loss_dec2 = (iden - c2).pow(2).sum()
            ret = loss_inv + self.lambd * (loss_dec1 + loss_dec2)
        elif loss_type == 'dcca':
            N = z1.shape[0]
            c = torch.mm(z1.T, z2)
            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)
            c = c / N
            c1 = c1 / N
            c2 = c2 / N
            loss_inv = - torch.diagonal(c).sum()
            ret = loss_inv
        elif loss_type == 'ntxent-uniform':
            l1 = self.semi_loss(z1, z2, indices, loss_type)
            l2 = self.semi_loss(z2, z1, indices, loss_type)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        elif loss_type == 'ntxent-align':
            l1 = self.semi_loss(z1, z2, indices, loss_type)
            l2 = self.semi_loss(z2, z1, indices, loss_type)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        return ret

class ContrastiveLearning(nn.Module):
    def __init__(self, args, data, device):
        super().__init__()
        self.dataset = args.dataset
        self.model = args.model
        self.epochs = args.epochs
        self.fmr = args.fmr
        self.edr = args.edr
        self.lambd = args.lambd
        self.batch = args.batch
        self.loss_type = args.loss_type
        self.data = data
        self.device = device
        self.num_class = int(self.data.y.max().item()) + 1 
        self.model = Model(self.data.num_features, args.hid_dim, args.out_dim, args.n_layers, args.tau, args.lambd, self.device, self.model, args.mlp_use)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr1, weight_decay=args.wd1)

        # if self.dataset == "ogbn-arxiv":
        #     self.s = lambda epoch: epoch / 1000 if epoch < 1000 else ( 1 + np.cos((epoch-1000) * np.pi / (self.epochs - 1000))) * 0.5
        #     self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.s)
        # if self.dataset in ['Swissroll','Moon','Circles']:
        #     self.logreg = LogReg(args.out_dim, 1)
        # else:
        self.logreg = LogReg(args.out_dim, self.num_class)
        self.logreg = self.logreg.to(self.device)
        self.opt = torch.optim.Adam(self.logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    def train(self):
        # loader = NeighborLoader(self.data, num_neighbors=[30] * 2, batch_size = self.batch, input_nodes=self.data.train_mask)
        for epoch in range(self.epochs):
            self.model.train()
            # for batch in loader:
                # print(batch.edge_index)
                # A = batch.edge_index[0]
                # print(A.unique().shape)
            self.optimizer.zero_grad()
            new_data1 = random_aug(self.data, self.fmr, self.edr) #batch
            new_data2 = random_aug(self.data, self.fmr, self.edr)
            new_data1 = new_data1.to(self.device)
            new_data2 = new_data2.to(self.device)
            u, v = self.model(new_data1, new_data2)
            loss = self.model.loss(u, v, self.batch, self.loss_type)
            loss.backward()
            self.optimizer.step()
            # if self.dataset == "ogbn-arxiv":
            #     self.scheduler.step()
            # print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))

    def uniformity(self, val_idx):
        new_data1 = random_aug(self.data,self.fmr,self.edr)
        new_data2 = random_aug(self.data,self.fmr,self.edr)
        z1 = F.normalize(self.model.get_embedding(new_data1).to(self.device)[val_idx])
        z2 = F.normalize(self.model.get_embedding(new_data2).to(self.device)[val_idx])
        sq_pdist1 = torch.pdist(z1, p=2).pow(2)
        sq_pdist2 = torch.pdist(z2, p=2).pow(2)
        l1 = sq_pdist1.mul(-2).exp().mean().log()
        l2 = sq_pdist2.mul(-2).exp().mean().log()
        return (l1+l2)/2

    def alignment(self, val_idx):
        new_data1 = random_aug(self.data,self.fmr,self.edr)
        new_data2 = random_aug(self.data,self.fmr,self.edr)
        z1 = F.normalize(self.model.get_embedding(new_data1).to(self.device)[val_idx])
        z2 = F.normalize(self.model.get_embedding(new_data2).to(self.device)[val_idx])
        return (z1-z2).norm(dim=1).pow(2).mean()

    def LinearEvaluation(self, train_idx, val_idx, test_idx):
        # self.model.eval()
        # if self.data == "ogbn-arxiv":
        #     evaluator = Evaluator(name='ogbn-arxiv')
        #     valid_acc = evaluator.eval({
        #                 'y_true': self.data.y[val_idx],
        #                 'y_pred': y_pred[val_idx],
        #                 })['acc']
        #     test_acc = evaluator.eval({
        #                 'y_true': data.y[split_idx['test']],
        #                 'y_pred': y_pred[split_idx['test']],
        #                 })['acc']
        # else:
        if self.dataset == "ogbn-arxiv":
            self.model = self.model.cpu()
            embeds = self.model.get_embedding(self.data)
            embeds = embeds.to(self.device)
        else:
            embeds = self.model.get_embedding(self.data.to(self.device))
            
        train_embs = embeds[train_idx]
        val_embs = embeds[val_idx]
        test_embs = embeds[test_idx]

        if self.dataset == "ogbn-arxiv":
            label = self.data.y.squeeze()
        else:
            label = self.data.y
        label = label.to(self.device)

        train_labels = label[train_idx]
        val_labels = label[val_idx]
        test_labels = label[test_idx]

        # calculate metric
        Lu = self.uniformity(val_idx)
        La = self.alignment(val_idx)

        loss_fn = nn.CrossEntropyLoss()       
        best_val_acc = 0
        eval_acc = 0

        for epoch in range(2000):
            self.logreg.train()
            self.opt.zero_grad()
            logits = self.logreg(train_embs)
            # preds = torch.argmax(logits, dim=1)
            # train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            self.opt.step()

            self.logreg.eval()
            with torch.no_grad():
                val_logits = self.logreg(val_embs)
                test_logits = self.logreg(test_embs)

                val_preds = torch.argmax(val_logits, dim=1)
                test_preds = torch.argmax(test_logits, dim=1)

                val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    if test_acc > eval_acc:
                        eval_acc = test_acc

        
            # print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))
        # print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
        return eval_acc, Lu, La
    

 # elif self.type == "bGRACE":
        #     z = F.relu(self.bnh(self.fc4(z)))
        #     h = self.bn(self.fc5(z))
        # elif self.type == "nonlinear":
        #     h = F.elu(self.fc3(z))
        # elif self.type == "linear":
        #     h = self.fc3(z)
        # elif self.method == "CLNR2":
        #     z = torch.vstack((z1,z2))
        #     h = (z - z.mean(0)) / z.std(0)
        #     h1, h2 = torch.split(h, [z1.shape[0],z1.shape[0]])
        # elif loss_type == "align":
        #     ret = self.semi_loss(h1, h2, indices, loss_type)
        # elif loss_type == "uniform":
        #     l1 = self.semi_loss(h1, h2, indices, loss_type)    
        #     l2 = self.semi_loss(h2, h1, indices, loss_type)    
        #     ret = ((l1 + l2) * 0.5)

# elif self.model == "CLNR":
#             z1 = (u - u.mean(0)) / u.std(0)
#             z2 = (v - v.mean(0)) / v.std(0)
#         elif self.model == "bCLNR":
#             z1 = self.bn(u)
#             z2 = self.bn(v)
#         elif self.model == 'dCLNR':
#             dbn1 = DBN(device=u.device, num_features=u.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
#             z1 = dbn1(u)
#             dbn2 = DBN(device=v.device, num_features=v.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
#             z2 = dbn2(v)
        # self.fc3 = nn.Linear(out_dim, out_dim)
        # # bgrace
        # self.fc4 = nn.Linear(out_dim, out_dim * 2)
        # self.fc5 = nn.Linear(out_dim * 2, out_dim)
        #self.bnh = nn.BatchNorm1d(out_dim * 2)
        # self.optimizer.zero_grad()
        #         new_data1 = random_aug(self.data, self.fmr, self.edr)
        #         new_data2 = random_aug(self.data, self.fmr, self.edr)
        #         new_data1 = new_data1.to(self.device)
        #         new_data2 = new_data2.to(self.device)
        #         u, v = self.model(new_data1, new_data2)   
        #         loss = self.model.loss(u, v, self.batch, self.loss_type)
        #         loss.backward()
        #         self.optimizer.step()
        #         print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
        # if k is not None:
        #     N = u.shape[0]
        #     indices = torch.LongTensor(random.sample(range(N), k))
        # else:
        #     indices = None