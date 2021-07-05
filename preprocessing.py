
import torch
import copy
import pandas as pd
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

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
