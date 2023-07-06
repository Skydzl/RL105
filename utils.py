import numpy as np
import torch

def to_onehot(index, length: int, device):
    if type(index) in [int, np.ndarray]:
        index = torch.tensor(index, dtype=int, device=device)
    if index.dim() > 1:
        res = torch.zeros(*index.shape[:-1], length, device=device)
    else:
        res = torch.zeros(length, device=device)
    res.scatter_(index.dim() - 1, index, 1)
    return res