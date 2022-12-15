import torch
from matplotlib import pyplot as plt

def visualize_uniformity(embs, labels):
    X = torch.nn.functional.normalize(embs, p=2.0, dim = 1)
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    scatter = plt.scatter(X[:,0],X[:,1],c=labels)
    scatter = plt.scatter(z[:, 0], z[:, 1], s=size, c=color, cmap="Set2")
    # produce a legend with the unique colors from the scatter
    legend1 = plt.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
    plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/uniformity.png')    
    # plt.show()
