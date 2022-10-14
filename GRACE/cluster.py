import torch
import numpy as np

from matplotlib import pyplot as plt
# import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pdb

# def visualize_umap(out, color, size=30, epoch=None, loss = None):
#     umap_2d = umap.UMAP(n_components=2, init="random", random_state=0)
#     z = umap_2d.fit_transform(out.detach().cpu().numpy())
#     plt.figure(figsize=(7,7))
#     plt.xticks([])
#     plt.yticks([])
#     scatter = plt.scatter(z[:, 0], z[:, 1], s=size, c=color, cmap="Set2")
#     # produce a legend with the unique colors from the scatter
#     legend1 = plt.legend(*scatter.legend_elements(),
#                     loc="lower left", title="Classes")
#     if epoch is not None and loss is not None:
#         plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=16)
#     plt.show()

def visualize_pca(out, color, pc1, pc2, path, model, size=30, epoch = None, loss = None):
    pca_4d = PCA(n_components=4)
    z = pca_4d.fit_transform(out.detach().cpu().numpy())
    evr = pca_4d.explained_variance_ratio_
    # pdb.set_trace()
    plt.figure(figsize=(7,7))
    plt.tick_params(axis='both', labelsize=15)
    scatter = plt.scatter(z[:, pc1-1], z[:, pc2-1], s=size, c=color, cmap="Set2")
    plt.xlabel("-".join(["PC",str(pc1)]), fontsize=20)
    plt.ylabel("-".join(["PC",str(pc2)]), fontsize=20)
    # produce a legend with the unique colors from the scatter
    legend1 = plt.legend(*scatter.legend_elements(),loc="lower left", title="Classes", prop={'size': 13} )
    plt.title("EVR:"+  str(round(evr[pc1],4)) + " " + "vs" + " " + str(round(evr[pc2],4)), fontsize = 20)
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=20)
    plt.savefig(path + "_" + model + "_" + 'cora_pc' + str(pc1) + str(pc2) + '.png')    
    # plt.show()

# def visualize_tsne(out, color, size=30, epoch=None, loss = None):
#     tsne_2d = TSNE(perplexity=30)
#     z = tsne_2d.fit_transform(out.detach().cpu().numpy())
#     plt.figure(figsize=(7,7))
#     plt.xticks([])
#     plt.yticks([])
#     scatter = plt.scatter(z[:, 0], z[:, 1], s=size, c=color, cmap="Set2")
#     # produce a legend with the unique colors from the scatter
#     legend1 = plt.legend(*scatter.legend_elements(),
#                     loc="lower left", title="Classes")
#     if epoch is not None and loss is not None:
#         plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=16)
#     plt.show()

