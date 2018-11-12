''' Controlled Parameter Sharing
'''
import operator
import functools

from torch import nn


class ContextVariable(nn.Module):
    ''' Context Variable Generator
    Generate a parameter with a given shape from `context` vector
    '''
    def __init__(self, context_size, shape, rank):
        super().__init__()
        self.size = functools.reduce(operator.mul, shape)
        self.shape = shape

        self.projection = nn.Linear(context_size, rank, bias=False)
        self.weights = nn.Linear(rank, self.size, bias=False)

    def forward(self, context):
        p = self.projection(context)
        return self.weights(p).reshape(self.shape)


class Context(nn.Module):
    ''' Module for Context
    Initialze the embedding for each context with N(0, 0.1^2)
    '''
    def __init__(self, contexts, context_size):
        super().__init__()
        self.context_embedding = nn.Embedding(contexts, context_size, 
                                              max_norm=1.0, norm_type=2)
        self.context_size = context_size
        for param in self.parameters():
            nn.init.normal_(param.data, std=0.1)

    def forward(self, context):
        return self.context_embedding(context)
