''' Neural Network Layers with Context Parameter Generator
'''
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

from . import cpg


class ContextModule(nn.Module):
    ''' Base class for all context modules
    '''


class ContextSequential(ContextModule, nn.Sequential):
    ''' nn.Sequential adapt for ContextModules
    '''
    def forward(self, ctx, input):
        ''' Forward through a list of layers sequentially,
        call layer with context when the layer is a ContextLayer
        '''
        for module in self._modules.values():
            if isinstance(module, ContextModule):
                input = module(ctx, input)
            else:
                input = module(input)
        return input


class Conv2d(ContextModule):
    ''' 2d Convolutional Layer
    '''
    context_size = None
    rank = None
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, **conv_params):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        h, w = kernel_size
        self.kernel = cpg.ContextVariable(self.context_size,
                                          (out_channels, in_channels, h, w),
                                          self.rank)
        if bias:
            self.bias = cpg.ContextVariable(self.context_size,
                                            (out_channels,),
                                            self.rank)
        else:
            self.bias = None
        self.conv_params = conv_params

    def forward(self, context, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias(context)

        return F.conv2d(x, self.kernel(context), bias=bias, **self.conv_params)


class ConvTranspose2d(ContextModule):
    ''' 2d Transposed Convolutional Layer
    '''
    context_size = None
    rank = None
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, **conv_params):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        h, w = kernel_size
        self.kernel = cpg.ContextVariable(self.context_size,
                                          (in_channels, out_channels, h, w),
                                          self.rank)
        if bias:
            self.bias = cpg.ContextVariable(self.context_size,
                                            (out_channels,),
                                            self.rank)
        else:
            self.bias = None
        self.conv_params = conv_params

    def forward(self, context, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias(context)

        return F.conv_transpose2d(x, self.kernel(context), bias=bias, **self.conv_params)
