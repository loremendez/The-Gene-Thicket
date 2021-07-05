import TCDF 
import torch
import pandas as pd
import numpy as np
import networkx as nx
import pylab
import copy
import matplotlib.pyplot as plt
import os
import sys
import pickle

def runTCDF(gem, tf_target_pairs=None, cuda=False, epochs=500, kernel_size=4, levels=0, lr=0.01, optimizername='Adam', seed=1, dilation_c=4, significance=0.8, log_interval=100):
    """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names."""

    links=pd.DataFrame()
    
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        if cuda==False:
            print("WARNING: You have a CUDA device, you should probably set cuda=True to speed up training.")
            
    links = pd.DataFrame()
    
    #if we have a previous list of TF-target genes
    if not tf_target_pairs is None:
        
        target_genes = list(tf_target_pairs.keys())
        
        for i, gene in enumerate(target_genes):
            
            #as a temporal Progress Bar
            if i % log_interval ==0 or i % len(target_genes) == 0 or i==0:
                percentage = (i/len(target_genes))*100
                print('Number of genes: {:2d} [{:.0f}%]'.format(i, percentage))
            #iterate through every gene
            if tfs_name != []:
                causes, realloss, scores_validated = TCDF.findcauses(target_name=gene, 
                                                                     tfs_name=tf_target_pairs[key], 
                                                                     gem=gem, 
                                                                     cuda=cuda, 
                                                                     epochs=epochs, 
                                                                     kernel_size=kernel_size, 
                                                                     layers=levels,
                                                                     lr=lr, 
                                                                     optimizername=optimizername, 
                                                                     seed=seed, 
                                                                     dilation_c=dilation_c, 
                                                                     significance=significance)
                temp_df = pd.DataFrame({'TF':causes, 'importance':scores_validated})
                temp_df['target'] = str(gene)
                links = links.append(temp_df)
                
    #to use the entire dataset to build the network
    else:
        
        target_genes = list(gem.columns)
        
        for i, gene in enumerate(target_genes):
            
            #as a temporal Progress Bar
            if i % log_interval ==0 or i % len(target_genes) == 0 or i==0:
                percentage = (i/len(target_genes))*100
                print('Number of genes: {:2d} [{:.0f}%]'.format(i, percentage))
            #iterate through every gene
            features = target_genes.copy()
            features.remove(gene)
            causes, realloss, scores_validated = TCDF.findcauses(target_name=gene,
                                                                 tfs_name=features, 
                                                                 gem=gem, 
                                                                 cuda=cuda, 
                                                                 epochs=epochs, 
                                                                 kernel_size=kernel_size, 
                                                                 layers=levels,
                                                                 lr=lr, 
                                                                 optimizername=optimizername, 
                                                                 seed=seed, 
                                                                 dilation_c=dilation_c, 
                                                                 significance=significance)
            temp_df = pd.DataFrame({'TF':causes, 'importance':scores_validated})
            temp_df['target'] = str(gene)
            links = links.append(temp_df)

    return links

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

# def evaluatedelay(extendedgtdelays, alldelays, TPs, receptivefield):
#     """Evaluates the delay discovery of TCDF by comparing the discovered time delays with the ground truth."""
#     zeros = 0
#     total = 0.
#     for i in range(len(TPs)):
#         tp=TPs[i]
#         discovereddelay = alldelays[tp]
#         gtdelays = extendedgtdelays[tp]
#         for d in gtdelays:
#             if d <= receptivefield:
#                 total+=1.
#                 error = d - discovereddelay
#                 if error == 0:
#                     zeros+=1
#             else:
#                 next
           
#     if zeros==0:
#         return 0.
#     else:
#         return zeros/float(total)

# def plotgraph(stringdatafile,alldelays,columns):
#     """Plots a temporal causal graph showing all discovered causal relationships annotated with the time delay between cause and effect."""
#     G = nx.DiGraph()
#     for c in columns:
#         G.add_node(c)
#     for pair in alldelays:
#         p1,p2 = pair
#         nodepair = (columns[p2], columns[p1])

#         G.add_edges_from([nodepair],weight=alldelays[pair])
    
#     edge_labels=dict([((u,v,),d['weight'])
#                     for u,v,d in G.edges(data=True)])
    
#     pos=nx.circular_layout(G)
#     nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
#     nx.draw(G,pos, node_color = 'white', edge_color='black',node_size=1000,with_labels = True)
#     ax = plt.gca()
#     ax.collections[0].set_edgecolor("#000000") 

#     pylab.show()
