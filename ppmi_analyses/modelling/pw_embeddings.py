import torch
import torch.nn as nn
import numpy as np

def broadcast(src, other, dim):
    # Source: torch_scatter
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


class SparseMaskedLinear_v2(nn.Module):
    """ Masked linear layer with sparse mask AND sparse weight matrix (faster and more memory efficient) """

    def __init__(self, in_features, out_features, sparse_mask, bias=True, device=None, dtype=None):
        """
        in_features: number of input features
        out_features: number of output features
        sparse_mask: torch tensor of shape (n_connections, 2), where indices[:, 0] index the input neurons
                     and indices[:, 1] index the output neurons
        """
        # Reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        self.sparse_mask = sparse_mask
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.nn.init.normal_(torch.empty((sparse_mask.shape[0]), **factory_kwargs)))  # Shape=(n_connections,)
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))

    def forward(self, input):
        # weight shape: (out_features, in_features)
        x = input[:, self.sparse_mask[:, 0]]  # Shape=(batch_size, n_connections)
        src = x * self.weight[None, :]  # Shape=(batch_size, n_connections)

        # Reduce via scatter sum
        out = torch.zeros((x.shape[0], self.out_features), dtype=x.dtype, device=x.device)
        index = broadcast(self.sparse_mask[:, 1], src, dim=-1)
        out = out.scatter_add_(dim=-1, index=index, src=src)
        if self.use_bias:
            out = out + self.bias
        return out


if __name__ == '__main__':
    in_features = 5
    out_features = 3
    x = torch.tensor(np.random.rand(19, in_features)).float()
    mask = np.random.binomial(n=1, p=0.5, size=(in_features, out_features))
    sparse_mask = torch.tensor(mask).nonzero()
    sparse_layer = SparseMaskedLinear_v2(in_features, out_features, sparse_mask)

    # Forward pass
    y_sparse = sparse_layer(x)