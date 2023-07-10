import collections
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch import nn


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出
        self.cnt = 0

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))
        self.cnt += 1

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        return transitions
        # state, action, reward, next_state, done = zip(*transitions)
        # return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
    
class MemoryQueue:
    "策略梯度用的 经验回放池"
    def __init__(self):
        self.buffer = collections.deque()
    
    def push(self, transitions):
        self.buffer.append(transitions)
        
    def sample(self):
        batch = list(self.buffer)
        return zip(*batch)
    
    def clear(self):
        self.buffer.clear()
        
    def __len__(self):
        return len(self.buffer)
    
    
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def plot_return_curve(return_list, env_name):
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()
    

def to_onehot(index, length: int, device):
    if type(index) in [int, np.ndarray]:
        index = torch.tensor(index, dtype=int, device=device)
    if index.dim() > 1:
        res = torch.zeros(*index.shape[:-1], length, device=device)
    else:
        res = torch.zeros(length, device=device)
    res.scatter_(index.dim() - 1, index, 1)
    return res


# def sequence_mask(X, mask_index, value=0):
#     """
#         X: (B, maxlen)
#         mask_index: (B, )
#     """
#     maxlen = X.shape[-1]
#     mask = torch.arange(maxlen, dtype=torch.float32)[None, :] < \
#         valid_lens[:, None]
#     X[~mask] = value
#     return X


# def action_list_to_mask(action_list, output_dim):
#     """
#     Parameters
#     ----------
#     action_list : TYPE list of list(index, continous, discrete)
#         DESCRIPTION. size: (num_actions, )
#     output_dim : TYPE 
#         DESCRIPTION. 输出维度，应和策略网络输出的维度一致，也就是project的数量

#     Returns
#     -------
#     mask_index size: (batch_size num_projects)
#         遮盖的位置为True，保留的地方为False
#     """
    
    
def masked_softmax(X, mask_index=None, value=-1e6):
    """
        遮盖softmax
        X: (num_projects, )
        mask_index: (num_actions, ) 不需要遮盖的index
    """
    if mask_index is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        mask_index = torch.tensor(list(
            set(torch.arange(X.shape[0]).numpy()) - set(mask_index.numpy())))
        X[mask_index] = value
    return nn.functional.softmax(X, dim=-1)
        






