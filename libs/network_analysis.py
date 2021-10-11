import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from cdlib import algorithms
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve

#The idea of the scores and plots was taken from CellOracle https://morris-lab.github.io/CellOracle.documentation/
#Code by Lorena Mendez

def plot_graph(df, type='links', figsize=(5,5), node_size=1000, font_size=8, savefig=None):
    """
    Plots the inferred GRN or the reference GRN from Beeline.

    :param df: pandas DataFrame. Either the links inferred by the CNN or the reference df from beeline.
    :param type: string. 'links' or 'ref', is it is the inferred links by the CNN or the reference from beeline, respectively.
    :param figsize: list of two numbers, (width, height).
    :param node_size: integer. Size of nodes.
    :param font_size: integer. Size of font.
    :param savefig: string. Path to save the figure.
    """

    #initialize the net
    DG = nx.DiGraph()

    if type =='links':
        #save weights
        weights = np.array(np.round(df['importance'],2))
        #add edges
        for i, u in enumerate(np.array(df)):
            DG.add_edge(u[0], u[2], weight=weights[i])

    elif type == 'ref':
        #we only have activation or repression
        weights = np.where(df.Type == '-', -1, 1)
        #add edges
        for i,u in enumerate(np.array(df)):
            DG.add_edge(u[0], u[1], weight=weights[i])
    else:
        raise NotImplementedError('the dataframe should be either links or reference')

    #blue for edges >0, red otherwise
    colors = np.where(weights<0,'r','b')

    plt.figure(figsize=figsize)
    #select type of graph
    pos = nx.circular_layout(DG)
    #draw edges
    nx.draw_networkx_edges(DG, pos, edge_color = colors, width=weights)
    #draw nodes
    nodes = nx.draw_networkx_nodes(DG, pos, node_size=node_size, node_color = np.array([(0,0,0,0)]))
    nodes.set_edgecolor('black')
    #draw labels
    nx.draw_networkx_labels(DG, pos, font_size=font_size, font_family="sans-serif")

    ax = plt.gca()
    ax.margins(0.02)
    plt.axis("off")
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, transparent=True)
    plt.show()


def scores(tf_target_df):
    """
    Returns a dataframe with network scores for each gene.

    :param tf_target_df: pandas DataFrame with two columns. The first column contains the name of the transcription
                         factor and the second column the name of the target gene.
    :return scores: pandas DataFrame with network scores for each gene.
    """
    #initialize the net
    DG = nx.DiGraph()

    for u in np.array(tf_target_df):
        DG.add_edge(u[0], u[1])

    scores = pd.DataFrame()
    
    genes = nx.in_degree_centrality(DG).keys()
        
    scores["genes"] = genes
    
    try:
        scores["eigenvector_centrality"] = [nx.eigenvector_centrality(DG)[gen] for gen in genes]
        scores["betweenness_centrality"] = [nx.betweenness_centrality(DG)[gen] for gen in genes]
        scores["closeness_centrality"] = [nx.closeness_centrality(DG)[gen] for gen in genes]
    except:
        pass
        
    scores["in_degree_centrality"] = [nx.in_degree_centrality(DG)[gen] for gen in genes]
    scores["out_degree_centrality"] = [nx.out_degree_centrality(DG)[gen] for gen in genes]
    scores["all_degree_centrality"] = [nx.degree_centrality(DG)[gen] for gen in genes]
    scores["degree_out"] = (scores["out_degree_centrality"] * (len(scores["genes"])-1)).astype(int)
    scores["degree_in"] = (scores["in_degree_centrality"] * (len(scores["genes"])-1)).astype(int)
    scores["degree_all"] = (scores["all_degree_centrality"] * (len(scores["genes"])-1)).astype(int)

    scores = scores.sort_values(by='genes').reset_index(drop=True)

    #add cartography analysis scores
    cartography_df = _cartography_analysis(tf_target_df)

    #merge both dataframes
    scores = pd.merge(scores, cartography_df, how="left", on=["genes"])

    return scores


def plot_scores(scores, n_genes=50, savefig=None):
    """
    Plots network scores for top n-th genes.

    :param scores: pandas DataFrame with network scores for each gene.
    :param n_genes: integer. number of genes to plot.
    :param savefig: string. Path to save the plot.
    """
    temp = scores.copy()
    temp.index = scores['genes']

    colnames = ["eigenvector_centrality", "betweenness_centrality", "closeness_centrality",
                "in_degree_centrality", "out_degree_centrality", "all_degree_centrality"]

    plt.figure(figsize=(15,8))

    for i, col in enumerate(colnames):

        df = temp[col].sort_values(ascending=False)
        df = df[:n_genes]

        plt.subplot(2,3, i+1)
        plt.barh(range(len(df)), df.values)
        plt.yticks(range(len(df)), df.index.values)
        plt.title(f" {col}")
        plt.gca().invert_yaxis()

        if not savefig is None:
            #os.makedirs(save, exist_ok=True)
            #path = os.path.join(save, f"ranked_values_in_{links.name}_{value}_{links.thread_number}_in_{cluster}.{settings['save_figure_as']}")
            plt.savefig(savefig, transparent=True)


def _cartography_analysis(tf_target_df):
    """
    Returns a dataframe with cartography classification for each gene.
    https://www.nature.com/articles/nature03288

    :param tf_target_df: pandas DataFrame with two columns. The first column contains the name of the transcription
                         factor and the second column the name of the target gene.
    :return cartography: pandas DataFrame with cartography scores and classification for each gene.
    """
    #initialize the net
    DG = nx.DiGraph()

    for u in np.array(tf_target_df):
        DG.add_edge(u[0], u[1])

    data = algorithms.walktrap(DG).to_node_community_map() #walktrap community detection

    membership_df = pd.DataFrame({'genes':[gen for gen in data.keys()],
                                  'membership':[data[gen][0] for gen in data.keys()]}) #community classification

    num_mod = max(membership_df['membership']) + 1 #number of communities
    #dataframe with degrees
    degree_df = pd.DataFrame(DG.degree)
    degree_df.columns = ['genes', 'degree']

    #within module degree
    z_score = pd.DataFrame()
    for i in np.arange(num_mod):
        temp_subgraph = DG.subgraph(membership_df[membership_df['membership']==i]['genes'])
        degree_sub_df = pd.DataFrame(temp_subgraph.degree)
        degree_sub_df.columns = ['genes', 'degree']
        temp_genes = np.array(degree_sub_df['genes'])
        temp_z = np.array((degree_sub_df['degree'] - np.mean(degree_sub_df['degree']))/  np.std(degree_sub_df['degree']))
        temp_df = pd.DataFrame({'genes':temp_genes, 'z_scores':temp_z})
        z_score = z_score.append(temp_df)

    #participation coefficient
    participation_coef = {}
    for gene in membership_df['genes']:
        mem_nei = membership_df[np.in1d(membership_df['genes'],[n for n in DG.neighbors(gene)])]
        deg_gene = degree_df[degree_df['genes']==gene]['degree']
        participation_coef[gene] = 1
        for s in np.arange(num_mod):
            deg_to_s = len(mem_nei[mem_nei['membership']==s])
            participation_coef[gene] = np.array(participation_coef[gene] - (deg_to_s / deg_gene)**2)[0]

    #summarize everything
    summary = z_score.copy()
    summary['membership'] = [membership_df[membership_df['genes'] == gene]['membership'].values[0] for gene in summary['genes']]
    summary['participation'] = [participation_coef[gene] for gene in summary['genes']]
    summary['classification'] = np.where((summary['z_scores']<2.5) & (summary['participation']<0.05), 'R1: Ultra-peripheral',
                                    np.where((summary['z_scores']<2.5) & (summary['participation']>=0.05), 'R2: Peripheral',
                                            np.where((summary['z_scores']<2.5) & (summary['participation']>=0.625),'R3: Non-hub connector',
                                                    np.where((summary['z_scores']<2.5) & (summary['participation']>=0.8),'R4: Non-hub kinless',
                                                            np.where((summary['z_scores']>=2.5) & (summary['participation']<0.3),'R5: Provincial hub',
                                                                    np.where((summary['z_scores']>=2.5) & (summary['participation']>=0.75), 'R7: Kinless hub', 'R6: Connector hub'))))))
    summary = summary.sort_values(by='genes').reset_index(drop=True)

    return summary

def plot_cartography(scores_df, highlight_genes=None, args_annot={}, args_line={}):
    """
    Plots Cartography Scores as in https://www.nature.com/articles/nature03288

    :param scores_df: scores dataframe. Output from scores function.
    :param highlight_genes: list of gene names to highlight.
    :param args_annot: args for the labels of highlighted genes.
    :param args_line: args for lines dividing the plane.
    """
    #default line args
    default = {"linestyle": "dashed", "alpha": 0.5, "c": "gray"}
    args_line.update(default)

    x, y = scores_df.participation, scores_df.z_scores
    
    z_min = min(np.min(scores_df['z_scores']),-2)
    z_max = max(np.max(scores_df['z_scores']), 8)

    plt.plot([-0.05, 1.05], [2.5, 2.5], **args_line)
    plt.plot([0.05, 0.05], [z_min, 2.5], **args_line)
    plt.plot([0.62, 0.62], [z_min, 2.5], **args_line)
    plt.plot([0.8, 0.8], [z_min, 2.5], **args_line)
    plt.plot([0.3, 0.3], [2.5, z_max + 0.5], **args_line)
    plt.plot([0.75, 0.75], [2.5, z_max + 0.5], **args_line)
    plt.scatter(x, y,  marker='o', edgecolor="lightgray", c="none", alpha=1)

    if not highlight_genes is None:
        for gen in highlight_genes:
            _highlight_gene(scores_df, gen, args_annot)

    plt.xlabel("Participation coefficient (P)")
    plt.ylabel("Whithin-module degree (z)")
    plt.title("Gene Cartography Analysis")

    plt.xlim([-0.1, 1.1])

def _highlight_gene(scores_df, gen, args_annot):
    """
    Highights a selected gene in the cartography plot.

    :param scores_df: scores dataframe. Output from scores function.
    :param gen: name of gene to highlight
    :param args_annot: args for the labels of highlighted genes.
    """
    x = scores_df[scores_df['genes']==gen]['participation']
    y = scores_df[scores_df['genes']==gen]['z_scores']
    plt.scatter(x, y, c="none", edgecolor="black")
    _annotate_gene(x, y, gen, args=args_annot)

def _annotate_gene(x, y, label, x_shift=0.05, y_shift=0.05, args={}):
    """
    Adds label to highlighted genes in the cartography plot. (From CellOracle)

    :param x: coordinate x of gene.
    :param y: coordinate y of gene.
    :param label: name of gene to highlight.
    :param x_shift: coordinate x for the label will be x + x_shift.
    :param y_shift: coordinate y for the label will be y + y_shift.
    :param args: args for the annotation.
    """

    #from CellOracle
    args_annot = {"size": 10, "color": "black"}
    args_annot.update(args)

    arrow_dict = {"width": 0.5, "headwidth": 0.5, "headlength": 1, "color": "black"}

    plt.annotate(label, xy=(x, y), xytext=(x+x_shift, y+y_shift), arrowprops=arrow_dict, **args_annot)

    
def evaluation(links, ref, tfs, target_genes):
    
    #possible edges
    possible_edges=[]
    
    for tf in tfs:
        for gene in target_genes:
            if gene != tf:
                possible_edges.append((tf, gene))
    
    #true edges
    true_edges = list(zip(ref['Gene1'].values,
                          ref['Gene2'].values))
    
    #predicted edges
    predicted_edges = list(zip(links['TF'].values,
                               links['target'].values))
    
    #compute true binary labels 
    y_true=[]
    for el in possible_edges:
        if el in true_edges:
            y_true.append(1)
        else:
            y_true.append(0)
    
    #compute predicted binary labels
    y_pred=[]
    for el in possible_edges:
        if el in predicted_edges:
            y_pred.append(1)
        else:
            y_pred.append(0)
    
    #compute metrics
    prec, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    
    fscore = f1_score(y_true, y_pred)
    AUCPR = auc(recall, prec)
    AUROC = auc(fpr, tpr)
    
    df = pd.DataFrame({'fscore':[fscore], 'AUCPR':[AUCPR], 'AUROC':[AUROC]})
    df = df.round(2)
    return df
    
# def full_graph(links):
#     DG = nx.DiGraph()
    
#     for i, u in enumerate(np.array(links)):
#         DG.add_edge(u[0], u[2])
    
#     for tf in links['TF'].unique():
#         for gene in links['target'].unique():
#             if nx.has_path(DG,tf,gene):
#                 DG.add_edge(tf,gene)
#     return DG

# def intersection(G,H):
#     R=nx.create_empty_copy(G)
#     if G.number_of_edges() <= H.number_of_edges():
#         edges=G.edges()
#         for e in edges:
#             if H.has_edge(*e):
#                 R.add_edge(*e)
#     else:
#         edges=H.edges()
#         for e in edges:
#             if G.has_edge(*e):
#                 R.add_edge(*e)
                
#     return R
    
# def evaluation(links, ref, possible_edges):
    
#     temp_df = links.copy()
#     temp_df['abs_weight'] = abs(temp_df['importance'])
#     temp_df = temp_df.sort_values(by='abs_weight', ascending=False).reset_index(drop=True)
    
#     #ground truth graph
#     TG = nx.DiGraph()
#     for i, u in enumerate(np.array(ref)):
#         TG.add_edge(u[0], u[1])
    
    
#     all_prec = []
#     all_recall = []
#     all_TPR = []
#     all_FPR = []
    
#     for row in np.arange(temp_df.shape[0]):
#         df = temp_df[:row+1]
        
#         #learnt graph
#         LG = nx.DiGraph()
#         for i, u in enumerate(np.array(df)):
#             LG.add_edge(u[0], u[2])
        
#         TP = intersection(TG,LG).number_of_edges() #true positives
#         FP = len(LG.edges() - TG.edges()) #false positives
#         FN = len(TG.edges() - LG.edges()) #false negatives
#         TN = possible_edges - (TP + FP + FN) #true negatives
#         precision =  TP /(TP + FP)
#         recall = TP /(FN + TP)
#         TPR = TP /(FN + TP) #sensitivity
#         FPR = FP /(FP + TN) #false positive rate
        
#         all_prec.append(precision)
#         all_recall.append(recall)
#         all_TPR.append(TPR)
#         all_FPR.append(FPR)
    
#     F1 = (2*TP)/(2*TP + 2*FN + FP) #F1 score
#     AUCPR = auc(all_recall, all_prec)
#     AUROC = auc(all_FPR, all_TPR)
    
    #full graph
    #FG = full_graph(links)
    
    ################ GROUND TRUTH GRAPH ##########################
#     TP = intersection(TG,LG).number_of_edges() #true positives
#     FP = len(LG.edges() - TG.edges()) #false positives
#     FN = len(TG.edges() - LG.edges()) #false negatives
#     TN = possible_edges - (TP + FP + FN) #true negatives
#     precision =  TP /(TP + FP)
#     recall = TP /(FN + TP)
#     TPR = TP /(FN + TP) #sensitivity
#     FPR = FP /(FP + TN) #false positive rate
#     F1 = (2*TP)/(2*TP + 2*FN + FP) #F1 score
    
    ################ FULL GRAPH #########################
#     TP_ = intersection(FG,LG).number_of_edges()
#     FP_ = len(LG.edges() - FG.edges()) #false positives
#     FN_ = len(FG.edges() - LG.edges()) #false negatives
#     TN_ = possible_edges - (TP_ + FP_ + FN_)
#     precision_ = TP_ /(TP_ + FP_)
#     recall_ = TP_ /(FN_ + TP_)
#     F1_ = (2*TP_)/(2*TP_ + 2*FN_ + FP_) #F1 score
    
#     df = pd.DataFrame({'precision':[precision], 'recall':[recall], 
#                        'TPR':[TPR], 'FPR':[FPR],'F1-score':[F1],
#                        'AUCPR':[AUCPR], 'AUROC':[AUROC]})
#     df = df.round(2)
#     return df