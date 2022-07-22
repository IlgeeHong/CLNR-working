setwd("/Users/ilgeehong/Desktop/SemGCon/DGI/results")
dgi_cora = read.csv("DGI_node_classification_Cora.csv")
mean(as.numeric(substr(dgi_cora$accuracy,8,13)))
sd(as.numeric(substr(dgi_cora$accuracy,8,13)))
