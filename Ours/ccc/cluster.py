import torch
from matplotlib import pyplot as plt

def visualize_uniformity(embs, labels, args, eval_acc):
    X = torch.nn.functional.normalize(embs, p=2.0, dim = 1).detach().cpu()
    y = labels.detach().cpu()
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    scatter = plt.scatter(X[:,0],X[:,1],c=y)
    plt.title("LE:"+  str(round(eval_acc,2)) , fontsize = 20)
    plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/uniformity'+ '_' + args.model+ '_' + '.png')    
    # plt.show()
