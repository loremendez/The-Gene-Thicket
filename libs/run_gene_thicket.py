import copy
import torch
import numpy as np
import pandas as pd
import networkx as nx
import libs.gene_thicket as gene_thicket
import matplotlib.pyplot as plt

def run_gene_thicket(gem, tf_target_pairs=None, cuda=False, epochs=1000, kernel_size=4, levels=1, lr=0.01, optimizername='Adam', seed=1111, dilation_c=4, significance=0.8, log_interval=500, patience=20):
    """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names."""
    
    #if cuda == True:
    #    if torch.cuda.is_available():
    #        cuda=True

    alldelays = dict()
    allreallosses=dict()
    links=pd.DataFrame()
    
    if not tf_target_pairs is None:
        target_genes = list(tf_target_pairs.keys())
        
        for i, gene in enumerate(target_genes):
            idx = gem.columns.get_loc(gene)
            tfs_name = tf_target_pairs[gene]
            if tfs_name != []:
                causes, causeswithdelay, realloss, scores, weights, scores_validated = gene_thicket.findcauses(target_name=gene, tfs_name=tfs_name, gem=gem, cuda=cuda, epochs=epochs, kernel_size=kernel_size, layers=levels, log_interval=log_interval, lr=lr, optimizername=optimizername, seed=seed, dilation_c=dilation_c, significance=significance, patience=patience)
                alldelays.update(causeswithdelay)
                allreallosses[idx]=realloss
                
                temp_df = pd.DataFrame({'TF':causes, 'importance':scores_validated})
                temp_df['target'] = str(gene)
                links = links.append(temp_df).reset_index(drop=True)
                torch.cuda.empty_cache()
    
    else:
        target_genes = list(gem)
        
        for i, gene in enumerate(target_genes):
            idx = gem.columns.get_loc(gene)
            features = target_genes.copy()
            features.remove(gene)
            causes, causeswithdelay, realloss, scores, weights, scores_validated = gene_thicket.findcauses(target_name=gene, tfs_name=features, gem=gem, cuda=cuda, epochs=epochs, kernel_size=kernel_size, layers=levels, log_interval=log_interval, lr=lr, optimizername=optimizername, seed=seed, dilation_c=dilation_c, significance=significance, patience=patience)
            
            temp_df = pd.DataFrame({'TF':causes, 'importance':scores_validated})
            temp_df['target'] = str(gene)
            links = links.append(temp_df).reset_index(drop=True)
            
            alldelays.update(causeswithdelay)
            allreallosses[idx]=realloss
            torch.cuda.empty_cache()
            
    #estimate correlation to infer sign of the links
    corr = np.zeros(links.shape[0])
    coefs = []
    for row in np.arange(links.shape[0]):
        tf = links['TF'][row]
        target = links['target'][row]
        coefs.append(np.corrcoef(gem[tf], gem[target])[0][1])
    corr += coefs
    
    links['importance'] = np.where(corr>0, 1, -1) * links['importance']

    #return allcauses, alldelays, allreallosses, allscores, target_genes, allweights, all_validated_scores, links
    return links, alldelays, allreallosses

# def evaluate(gtfile, validatedcauses, columns):
#     """Evaluates the results of TCDF by comparing it to the ground truth graph, and calculating precision, recall and F1-score. F1'-score, precision' and recall' include indirect causal relationships."""
#     extendedgtdelays, readgt, extendedreadgt = getextendeddelays(gtfile, columns)
#     FP=0
#     FPdirect=0
#     TPdirect=0
#     TP=0
#     FN=0
#     FPs = []
#     FPsdirect = []
#     TPsdirect = []
#     TPs = []
#     FNs = []
#     for key in readgt:
#         for v in validatedcauses[key]:
#             if v not in extendedreadgt[key]:
#                 FP+=1
#                 FPs.append((key,v))
#             else:
#                 TP+=1
#                 TPs.append((key,v))
#             if v not in readgt[key]:
#                 FPdirect+=1
#                 FPsdirect.append((key,v))
#             else:
#                 TPdirect+=1
#                 TPsdirect.append((key,v))
#         for v in readgt[key]:
#             if v not in validatedcauses[key]:
#                 FN+=1
#                 FNs.append((key, v))
          
#     print("Total False Positives': ", FP)
#     print("Total True Positives': ", TP)
#     print("Total False Negatives: ", FN)
#     print("Total Direct False Positives: ", FPdirect)
#     print("Total Direct True Positives: ", TPdirect)
#     print("TPs': ", TPs)
#     print("FPs': ", FPs)
#     print("TPs direct: ", TPsdirect)
#     print("FPs direct: ", FPsdirect)
#     print("FNs: ", FNs)
#     precision = recall = 0.

#     if float(TP+FP)>0:
#         precision = TP / float(TP+FP)
#     print("Precision': ", precision)
#     if float(TP + FN)>0:
#         recall = TP / float(TP + FN)
#     print("Recall': ", recall)
#     if (precision + recall) > 0:
#         F1 = 2 * (precision * recall) / (precision + recall)
#     else:
#         F1 = 0.
#     print("F1' score: ", F1,"(includes direct and indirect causal relationships)")

#     precision = recall = 0.
#     if float(TPdirect+FPdirect)>0:
#         precision = TPdirect / float(TPdirect+FPdirect)
#     print("Precision: ", precision)
#     if float(TPdirect + FN)>0:
#         recall = TPdirect / float(TPdirect + FN)
#     print("Recall: ", recall)
#     if (precision + recall) > 0:
#         F1direct = 2 * (precision * recall) / (precision + recall)
#     else:
#         F1direct = 0.
#     print("F1 score: ", F1direct,"(includes only direct causal relationships)")
#     return FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct


# def getextendeddelays(gtfile, columns):
#     """Collects the total delay of indirect causal relationships."""
#     gtdata = pd.read_csv(gtfile, header=None)

#     readgt=dict()
#     effects = gtdata[1]
#     causes = gtdata[0]
#     delays = gtdata[2]
#     gtnrrelations = 0
#     pairdelays = dict()
#     for k in range(len(columns)):
#         readgt[k]=[]
#     for i in range(len(effects)):
#         key=effects[i]
#         value=causes[i]
#         readgt[key].append(value)
#         pairdelays[(key, value)]=delays[i]
#         gtnrrelations+=1
    
#     g = nx.DiGraph()
#     g.add_nodes_from(readgt.keys())
#     for e in readgt:
#         cs = readgt[e]
#         for c in cs:
#             g.add_edge(c, e)

#     extendedreadgt = copy.deepcopy(readgt)
    
#     for c1 in range(len(columns)):
#         for c2 in range(len(columns)):
#             paths = list(nx.all_simple_paths(g, c1, c2, cutoff=2)) #indirect path max length 3, no cycles
            
#             if len(paths)>0:
#                 for path in paths:
#                     for p in path[:-1]:
#                         if p not in extendedreadgt[path[-1]]:
#                             extendedreadgt[path[-1]].append(p)
                            
#     extendedgtdelays = dict()
#     for effect in extendedreadgt:
#         causes = extendedreadgt[effect]
#         for cause in causes:
#             if (effect, cause) in pairdelays:
#                 delay = pairdelays[(effect, cause)]
#                 extendedgtdelays[(effect, cause)]=[delay]
#             else:
#                 #find extended delay
#                 paths = list(nx.all_simple_paths(g, cause, effect, cutoff=2)) #indirect path max length 3, no cycles
#                 extendedgtdelays[(effect, cause)]=[]
#                 for p in paths:
#                     delay=0
#                     for i in range(len(p)-1):
#                         delay+=pairdelays[(p[i+1], p[i])]
#                     extendedgtdelays[(effect, cause)].append(delay)

#     return extendedgtdelays, readgt, extendedreadgt