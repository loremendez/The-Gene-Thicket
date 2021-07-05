import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable

class Chomp1d(nn.Module):
    """PyTorch does not offer native support for causal convolutions, so it is implemented (with some inefficiency) by simply using a standard convolution with zero padding on both sides, and chopping off the end of the sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class FirstBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(FirstBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding, 
                               dilation=dilation, 
                               groups=n_outputs)
        nn.init.kaiming_normal(self.conv1.weight)
        
        self.net = nn.Sequential(self.conv1, 
                                 Chomp1d(padding),
                                 nn.PReLU(n_inputs))

    def forward(self, x):
        return self.net(x)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding, 
                               dilation=dilation, 
                               groups=n_outputs)
        nn.init.kaiming_normal(self.conv1.weight)
        
        self.net = nn.Sequential(self.conv1, Chomp1d(padding))
        self.relu = nn.PReLU(n_inputs)

    def forward(self, x):
        out = self.net(x)
        return self.relu(out+x) #residual connection

class LastBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(LastBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding, 
                               dilation=dilation, 
                               groups=n_outputs)
        nn.init.kaiming_normal(self.conv1.weight)
        
        self.net = nn.Sequential(self.conv1, Chomp1d(padding))
        self.linear = nn.Linear(n_inputs, n_inputs)

    def forward(self, x):
        out = self.net(x)
        return self.linear(out.transpose(1,2)+x.transpose(1,2)).transpose(1,2) #residual connection

class DepthwiseNet(nn.Module):
    def __init__(self, num_inputs, num_levels, kernel_size=2, dilation_c=2):
        super(DepthwiseNet, self).__init__()
        layers = []
        in_channels = num_inputs
        out_channels = num_inputs
        for l in range(num_levels):
            dilation_size = dilation_c ** l
            if l==0:
                  layers += [FirstBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size-1) * dilation_size)]

            elif l==num_levels-1:
                layers+=[LastBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)