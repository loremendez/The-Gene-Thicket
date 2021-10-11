import numpy as np
import pandas as pd
import seaborn as sns
import scvelo as scv
import anndata
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def pseudotime_to_velocity(adata, time_key='pseudotime'):
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
        time.append((adata.obs[time_key][neigh] - adata.obs[time_key][i]).values) #time differences
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

def random_walks(adata, num_walks=100, time_key='pseudotime', seed=0, n_steps=100, n_neighbors=30, starting_cells=None):
    """
    Generates different gene expression matrices using random walks. 
    
    :params adata: AnnData.
    :params num_walks: integer. Number of random walks (dataframes).
    :params time_key: string. Name of the column in adata.obs that contains the time. 
    :params seed: random seed.
    :params n_steps: integer. Number of steps in the random walk.
    :params n_neighbors: integer. Number of neighbors in which the cell can transition.
    :params starting_cells: list of integers. Number of cells from which the random walk will begin.
    :returns gem: dictionary with random-walk dataframes sorted by time.
    """
    np.random.seed(seed)
    
    n_cells = adata.shape[0]
    
    #choose starting cells
    if starting_cells is None:
        starting_cells = np.random.choice(adata.shape[0], num_walks, replace=False)
    
    #generate different expression matrices and sort them according to time
    gem={}
    for i in np.arange(num_walks):
        cells = scv.utils.get_cell_transitions(adata, n_steps=n_steps, n_neighbors=n_neighbors, starting_cell=starting_cells[i])
        bool_var = np.isin(np.arange(n_cells), cells)
        gem_temp = adata[bool_var]#.to_df()
        p_sort = gem_temp.obs.sort_values(by=time_key)
        p_cells = list(p_sort[p_sort[time_key].isna() == False].index) 
        gem[i] = gem_temp.to_df().loc[p_cells] #datasets are sorted by pseudotime
    
    return gem
    
