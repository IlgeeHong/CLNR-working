setwd("/Users/ilgeehong/Desktop/SemGCon/Ours/results")
selfgcon_cora = read.csv("SelfGCon_node_classification_Cora.csv")
mean(as.numeric(substr(selfgcon_cora$accuracy,8,13)))
sd(as.numeric(substr(selfgcon_cora$accuracy,8,13)))

selfgcon_citeseer = read.csv("SelfGCon_node_classification_CiteSeer.csv")
mean(as.numeric(substr(selfgcon_citeseer$accuracy,8,13)))
sd(as.numeric(substr(selfgcon_citeseer$accuracy,8,13)))

selfgcon_pubmed = read.csv("SelfGCon_node_classification_PubMed.csv")
mean(as.numeric(substr(selfgcon_pubmed$accuracy,8,13)))
sd(as.numeric(substr(selfgcon_pubmed$accuracy,8,13)))