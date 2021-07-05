import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model import ADDSTCN
import random
import pandas as pd
import numpy as np
import heapq
import copy
import os
import sys

def preparedata(target, tfs, gem):
    '''
    Transforms target and features into PyTorch tensors.
    
    :param target: string. Name of the target gene.
    :param tfs: list of strings. List with putative transcription factors corresponding to target gene.  
    :param gem: pandas DataFrame. Gene Expression Matrix. 
    :return x: pytorch tensor of size [1, tfs, timesteps]. Transcription factor time series.
    :return y: pytorch tensor of size [1, timesteps, 1]. Target gene time series.
    '''
    df_y = gem.copy(deep=True)[[target]]
    df_x = gem.copy(deep=True)[tfs]
    data_x = df_x.values.astype('float32').transpose()
    data_y = df_y.values.astype('float32').transpose()
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    x, y = Variable(data_x), Variable(data_y)
    
    x = x.unsqueeze(0).contiguous()
    y = y.unsqueeze(2).contiguous()
    
    return x, y


def train(traindata, traintarget, model, optimizer):
    '''
    Trains model for one epoch.
    
    :param traindata: pytorch tensor of size [1, tfs, timesteps].
    :param traintarget: pytorch tensor of size [1, timesteps, 1].
    :param model: pytorch model. Neural network architecture.
    :param optimizer: pytorch optimizer.
    :return attentionscores: pytorch tensor of size [tfs, 1].
    :return loss: pytorch tensor of size []. One Loss function value after one iteration.
    '''
    
    model.train()
    x, y = traindata[0:1], traintarget[0:1]
        
    optimizer.zero_grad()
    output = model(x)

    attentionscores = model.fs_attention
    
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()

    return attentionscores.data, loss

def findcauses(target_name, tfs_name, gem, cuda, epochs, kernel_size, layers, lr, optimizername, seed, dilation_c, significance):
    '''
    Discovers potential causes of one target time series, validates them through permutation and gets the corresponding time delays (to be implemented).
    
    :param target_name: string. Name of target gene.
    :param tfs_name: list of strings. List cointaing the name of all putative transcription factors associated with the target gene.
    :param gem: pandas DataFrame of size [time steps, genes] containing all the gene and tfs time series. 
    :param cuda: boolean. To use GPU or not.
    :param epochs: positive integer. Number of epochs to train the model.
    :param kernel_size: positive integer. size of kernel.
    :param layers: integer >= 0. Number of blocks, between first and last block. 'Depth of network'.
    :param lr: real number >0. Learning rate.
    :param optimizername: string. Name of optimizer.
    :param seed: integer. Random state.
    :param dilation_c: positive integer. Dilation constant, it should be equal to kernel_size.
    :param significance: real number in [0,1]. Proportion of the loss that needs to be overcomed, while permutating a feature to know if it is valid or not. 
    :return names_validated: list of tfs that are putative causes of target gene.
    :return realloss: real number. Value of the loss function at the end of the training.
    :return scores_validated: list of attention scores corresponding to the validated transcription factors.
    '''

    torch.manual_seed(seed)
    
    X_train, Y_train = preparedata(target_name, tfs_name, gem)

    input_channels = X_train.size()[1]
       
    model = ADDSTCN(input_channels, layers, kernel_size=kernel_size, cuda=cuda, dilation_c=dilation_c)
    
    if cuda:
        model.cuda()
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
    
    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)    
    
    scores, firstloss = train(X_train, Y_train, model, optimizer)
    
    firstloss = firstloss.cpu().data.item()
    
    for ep in range(2, epochs+1):
        scores, realloss = train(X_train, Y_train, model, optimizer)
    realloss = realloss.cpu().data.item()
    
    s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
    indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy())
    
    validated = list(copy.deepcopy(indices))
    
    #Apply PIVM (permutes the values) to check if potential cause is true cause
    for idx in indices:
        random.seed(seed)
        X_test2 = X_train.clone().cpu().numpy()
        random.shuffle(X_test2[:,idx,:][0])
        shuffled = torch.from_numpy(X_test2)
        if cuda:
            shuffled=shuffled.cuda()
        model.eval()
        output = model(shuffled)
        testloss = F.mse_loss(output, Y_train)
        testloss = testloss.cpu().data.item()
        
        diff = firstloss-realloss
        testdiff = firstloss-testloss

        if testdiff>(diff*significance): 
            validated.remove(idx) 
            
    names_validated = list(tfs_name[i] for i in validated)
    scores_all = scores.view(-1).cpu().detach().numpy().tolist()
    scores_validated = list(scores_all[i] for i in validated) 
    
    return names_validated, realloss, scores_validated





