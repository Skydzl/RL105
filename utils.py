import collections
import deuqe
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

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