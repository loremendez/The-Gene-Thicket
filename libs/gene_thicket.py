import torch
import random
import pandas as pd
import numpy as np
import heapq
import copy
from libs.causal_cnn import LaRed
from libs.earlystop import EarlyStopping

def preparedata(target_name, tfs_name, gem):
    '''
    Transforms target and features into PyTorch tensors.

    :param target: string. Name of the target gene.
    :param tfs: list of strings. List with putative transcription factors corresponding to target gene.
    :param gem: pandas DataFrame. Gene Expression Matrix.
    :return x: pytorch tensor of size [1, tfs, timesteps]. Transcription factor time series.
    :return y: pytorch tensor of size [1, timesteps, 1]. Target gene time series.
    '''
    df_y = gem.copy(deep=True)[[target_name]]
    df_x = gem.copy(deep=True)[tfs_name]
    
    df_xshift = df_x.copy(deep=True).shift(periods=1, axis=0).fillna(0.)
    
    data_x = df_xshift.values.astype('float32').transpose()
    data_y = df_y.values.astype('float32').transpose()
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    x, y = torch.autograd.Variable(data_x), torch.autograd.Variable(data_y)
    x = x.unsqueeze(0).contiguous()
    y = y.unsqueeze(2).contiguous()

    return x, y


def train(epoch, traindata, traintarget, modelname, optimizer, log_interval, epochs, proportion=0.8):
    '''
    Trains model for one epoch, using the first 80% of the data (sorted in chronological order). 

    :param traindata: pytorch tensor of size [1, tfs, timesteps].
    :param traintarget: pytorch tensor of size [1, timesteps, 1].
    :param model: pytorch model. Neural network architecture.
    :param optimizer: pytorch optimizer.
    :return attentionscores: pytorch tensor of size [tfs, 1].
    :return loss: pytorch tensor of size []. One Loss function value after one iteration.
    '''

    x, y = traindata[0:1], traintarget[0:1]

    index_train = int(x.shape[2] * proportion) #use 80% to train

    #divide data set in train and validation
    x_train, y_train = x[:,:,:index_train], y[:,:index_train,:]
    x_val, y_val = x[:,:,index_train:], y[:,index_train:,:]

    modelname.train()
    optimizer.zero_grad()

    #training set
    output = modelname(x_train)
    attentionscores = modelname.fs_attention
    loss = torch.nn.functional.mse_loss(output, y_train)
    loss.backward()
    optimizer.step()
    train_loss = loss.data.item()

    #validation set
    modelname.eval()
    output = modelname(x_val)
    loss = torch.nn.functional.mse_loss(output, y_val)
    valid_loss = loss.data.item()

    epochpercentage = (epoch/float(epochs))*100
    if epoch % log_interval ==0 or epoch % epochs == 0 or epoch==1:
        print('Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}'.format(epoch, epochpercentage, loss))

    return attentionscores.data, train_loss, valid_loss

def train_full(traindata, traintarget, modelname, optimizer):
    '''
    Trains model for one epoch using 100% of the data.

    :param traindata: pytorch tensor of size [1, tfs, timesteps].
    :param traintarget: pytorch tensor of size [1, timesteps, 1].
    :param model: pytorch model. Neural network architecture.
    :param optimizer: pytorch optimizer.
    :return attentionscores: pytorch tensor of size [tfs, 1].
    :return loss: pytorch tensor of size []. One Loss function value after one iteration.
    '''

    x, y = traindata[0:1], traintarget[0:1]

    modelname.train()
    optimizer.zero_grad()

    #training set
    output = modelname(x)
    attentionscores = modelname.fs_attention
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    realloss = loss.data.item()

    return attentionscores.data, realloss

def findcauses(target_name, tfs_name, gem, cuda=False, epochs=1000, kernel_size=4, layers=1,
               log_interval=500, lr=0.01, optimizername='Adam', seed=1111, dilation_c=4, significance=0.8, patience=20):
    '''
    Discovers potential causes of one target time series, validates them through permutation and gets the corresponding time delays (to be implemented).

    :param target_name: string. Name of target gene.
    :param tfs_name: list of strings. List cointaing the name of all putative transcription factors associated with the target gene.
    :param gems: dictionary with 1 or more pandas DataFrame of size [time steps, genes] containing all the gene and tfs time series.
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

    print("\n", "Analysis started for target: ", target_name)
    torch.manual_seed(seed)

    X_train, Y_train = preparedata(target_name, tfs_name, gem)

    input_channels = X_train.size()[1]

    targetidx = gem.columns.get_loc(target_name)

    model = LaRed(input_channels, layers, kernel_size=kernel_size, cuda=cuda, dilation_c=dilation_c)
    if cuda:
        model.cuda()
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()

    optimizer = getattr(torch.optim, optimizername)(model.parameters(), lr=lr)

    #save first loss to validate causes later
    _, firstloss, _ = train(1, X_train, Y_train, model, optimizer, log_interval, epochs, proportion=0.8)
    #firstloss = firstloss.cpu().data.item()

    #training
    early_stopping = EarlyStopping(patience=patience) #add patience as hyperparameter
    #train with a percentage of the data
    for ep in range(2, epochs+1):
        scores, realloss, validloss = train(ep, X_train, Y_train, model, optimizer, log_interval, epochs, proportion=0.8)
        early_stopping(validloss, model)
        
        if early_stopping.early_stop:
            print('Early stopping')
            break
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    #train for a few iterations using the whole data set 
    for ep in range(1,int(epochs*0.1)):
        scores, realloss = train_full(X_train, Y_train, model, optimizer)
          
    #scores = model.fs_attention
    #realloss = realloss.cpu().data.item()

    s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
    indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy())

    #attention interpretation to find tau: the threshold that distinguishes potential causes from non-causal time series
    if len(s)<=5:
        potentials = []
        for i in indices:
            if scores[i]>1.:
                potentials.append(i)
    else:
        potentials = []
        gaps = []
        for i in range(len(s)-1):
            if s[i]<1.: #tau should be greater or equal to 1, so only consider scores >= 1
                break
            gap = s[i]-s[i+1]
            gaps.append(gap)
        sortgaps = sorted(gaps, reverse=True)

        for i in range(0, len(gaps)):
            largestgap = sortgaps[i]
            index = gaps.index(largestgap)
            ind = -1
            if index>0:
                ind=index
                break
            
#             if index<((len(s)-1)/2): #gap should be in first half
#                 if index>0:
#                     ind=index #gap should have index > 0, except if second score <1
#                     break
        if ind<0:
            ind = 0

        potentials = indices[:ind+1].tolist()
    #print("Potential causes: ", potentials)
    validated = copy.deepcopy(potentials)

    #Apply PIVM (permutes the values) to check if potential cause is true cause
    for idx in potentials:
        random.seed(seed)
        X_test2 = X_train.clone().cpu().numpy()
        random.shuffle(X_test2[:,idx,:][0])
        shuffled = torch.from_numpy(X_test2)
        if cuda:
            shuffled=shuffled.cuda()
        model.eval()
        output = model(shuffled)
        testloss = torch.nn.functional.mse_loss(output, Y_train)
        testloss = testloss.cpu().data.item()

        diff = firstloss-realloss
        testdiff = firstloss-testloss

        if testdiff>(diff*significance):
            validated.remove(idx)

    weights = []

    scores_all = scores.view(-1).cpu().detach().numpy().tolist()
    
    
    names_validated = list(tfs_name[i] for i in validated)
    scores_validated = list(scores_all[i] for i in validated)
    
    causeswithdelay=[]
    
    #torch.save(model.state_dict(), './saved_models_pancreas/' + target_name + '.pt')

    return names_validated, causeswithdelay, realloss, scores_all, weights, scores_validated
