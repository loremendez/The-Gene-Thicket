import numpy as np
import pandas as pd
import seaborn as sns
import scvelo as scv
import anndata
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def pseudotime_to_velocity(adata):
    """
    Estimates velocity from pseudotime data.

    :params adata: anndata with gene expression, pseudotime in adata.obs,
                   and already performed scv.pp.neighbors.
    :returns : modified anndata with 'velocity' and 'velocity_graph' 
    """
    #assess if pp.neighbors was already performed
    if not 'neighbors' in adata.uns.keys():
        raise NotImplementedError("run scv.pp.neighbors first")

    n_neighbors = adata.uns['neighbors']['params']['n_neighbors'] #number of neighbors
    n_cells = adata.shape[0] #number of cells
    n_genes = adata.shape[1] #number of genes

    #row and column indices
    row_ind = adata.obsp['distances'].tocoo().row
    col_ind = adata.obsp['distances'].tocoo().col

    #estimate time differences
    time=[]
    for i in np.arange(n_cells):
        neigh = col_ind[row_ind == i] #list of neighbors
        time.append((adata.obs['pseudotime'][neigh] - adata.obs['pseudotime'][i]).values) #time differences
    time = np.array(time).flatten()

    #estimate weights
    time_diff = csr_matrix((time, (row_ind, col_ind)))
    w_normalized = normalize(time_diff.power(2), norm='l1', axis=1) #use squared time difference as weight

    #save velocity graph (weights matrix)
    adata.uns['velocity_graph'] = w_normalized #save it as velocity graph to later project into the embedding

    #estimate velos
    velocity = pd.DataFrame()
    for i in np.arange(n_cells):
        neigh = col_ind[row_ind == i] #list of neighbors
        displacement = (np.array(adata[i].to_df()) - adata[neigh].to_df()) #displacement to each neighbor
        time_denom = np.repeat((1/time_diff.data[row_ind==i]),n_genes).reshape((n_neighbors-1, n_genes)) #time reciprocal
        temp_velos = displacement/time_denom
        velocity = velocity.append(np.matmul(w_normalized.data[row_ind==i], temp_velos), ignore_index=True) #weighted velocity

    #save velocity estimation
    adata.layers["velocity"] = velocity
