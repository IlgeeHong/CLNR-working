setwd("/Users/ilgeehong/Desktop/SemGCon/Ours/results")
#selfgcon_cora = read.csv("SelfGCon_node_classification_Cora.csv")
#mean(as.numeric(substr(selfgcon_cora$accuracy,8,13)))
#sd(as.numeric(substr(selfgcon_cora$accuracy,8,13)))

#selfgcon_citeseer = read.csv("SelfGCon_node_classification_CiteSeer.csv")
#mean(as.numeric(substr(selfgcon_citeseer$accuracy,8,13)))
#sd(as.numeric(substr(selfgcon_citeseer$accuracy,8,13)))

#selfgcon_pubmed = read.csv("SelfGCon_node_classification_PubMed.csv")
#mean(as.numeric(substr(selfgcon_pubmed$accuracy,8,13)))
#sd(as.numeric(substr(selfgcon_pubmed$accuracy,8,13)))

cora = read.csv("Slurm_SelfGCon_Cora.csv")
cora[order(cora$val_acc, decreasing = TRUE),]

citeseer = read.csv("Slurm_SelfGCon_CiteSeer.csv")
citeseer[order(citeseer$val_acc, decreasing = TRUE),]