import copy
import torch
import numpy as np
import pandas as pd
import networkx as nx
import libs_TCDF.TCDF as TCDF
import matplotlib.pyplot as plt

def runTCDF(gem, tf_target_pairs=None, cuda=False, epochs=1000, kernel_size=4, levels=1, lr=0.01, optimizername='Adam', seed=1111, dilation_c=4, significance=0.8, log_interval=500):
    """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names."""
    
    if torch.cuda.is_available():
        cuda=True

    alldelays = dict()
    allreallosses=dict()
    links=pd.DataFrame()
    
    if not tf_target_pairs is None:
        target_genes = list(tf_target_pairs.keys())
        
        for i, gene in enumerate(target_genes):
            idx = gem.columns.get_loc(gene)
            if tfs_name != []:
                causes, causeswithdelay, realloss, scores, weights, scores_validated = TCDF.findcauses(target_name=gene, tfs_name=tf_target_pairs[gene], gem=gem, cuda=cuda, epochs=epochs, kernel_size=kernel_size, layers=levels, log_interval=log_interval, lr=lr, optimizername=optimizername, seed=seed, dilation_c=dilation_c, significance=significance)
                alldelays.update(causeswithdelay)
                allreallosses[idx]=realloss
                
                temp_df = pd.DataFrame({'TF':causes, 'importance':scores_validated})
                temp_df['target'] = str(gene)
                links = links.append(temp_df).reset_index(drop=True)
    
    else:
        target_genes = list(gem)
        
        for i, gene in enumerate(target_genes):
            idx = gem.columns.get_loc(gene)
            features = target_genes.copy()
            #features.remove(gene)
            causes, causeswithdelay, realloss, scores, weights, scores_validated = TCDF.findcauses(target_name=gene, tfs_name=features, gem=gem, cuda=cuda, epochs=epochs, kernel_size=kernel_size, layers=levels, log_interval=log_interval, lr=lr, optimizername=optimizername, seed=seed, dilation_c=dilation_c, significance=significance)
            
            temp_df = pd.DataFrame({'TF':causes, 'importance':scores_validated})
            temp_df['target'] = str(gene)
            links = links.append(temp_df).reset_index(drop=True)
            
            alldelays.update(causeswithdelay)
            allreallosses[idx]=realloss
            
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

