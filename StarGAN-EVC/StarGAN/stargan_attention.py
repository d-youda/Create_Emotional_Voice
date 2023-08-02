"""
average_weighted_attentions.py

Author: Max Elliott

Implementation of the average weighted attention layers proposed in
ATTENTION-AUGMENTED END-TO-END MULTI-TASK LEARNING FOR EMOTION PREDICTION FROM
SPEECH by Zixing Zhang.
"""

import torch
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    def __init__(self, vector_size):
        super(Attention, self).__init__()
        self.vector_size = vector_size
        self.weights = nn.Parameter(torch.rand(self.vector_size, 1, requires_grad=True)/np.sqrt(self.vector_size))

    def forward(self, x):
        '''
        x.size() = (Batch, max_seq_len, n_feats)
        '''
        original_sizes = x.size()
        x = x.contiguous().view(original_sizes[0]*original_sizes[1],1) #contiguous : 순서대로 자료 저장됨

        x_dot_w = x.mm(self.weight)

        x_dot_w = x_dot_w.view(original_sizes[0], original_sizes[1])

        softmax = nn.Softmax(dim=1)
        alphas = softmax(x_dot_w)
        alphas = alphas.view(-1,1) #alphas(n,1)

        x = x.mul(alphas)
        x = x.view(original_sizes)
        x = torch.sum(x, dim=1)

        return x


