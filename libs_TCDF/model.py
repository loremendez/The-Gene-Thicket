import torch
import numpy as np
from libs_good.depthwise import DepthwiseNet

class ADDSTCN(torch.nn.Module):
    def __init__(self, input_size, num_levels, kernel_size, cuda, dilation_c):
        super(ADDSTCN, self).__init__()

        self.dwn = DepthwiseNet(input_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = torch.nn.Conv1d(input_size, 1, 1)

        self._attention = torch.ones(input_size,1)
        self._attention = torch.autograd.Variable(self._attention, requires_grad=False)

        self.fs_attention = torch.nn.Parameter(self._attention.data)
        
        if cuda:
            self.dwn = self.dwn.cuda()
            self.pointwise = self.pointwise.cuda()
            self._attention = self._attention.cuda()    
        
    def forward(self, x):
        y1=self.dwn(x*torch.nn.functional.softmax(self.fs_attention, dim=0))
        y1 = self.pointwise(y1) 
        return y1.transpose(1,2)