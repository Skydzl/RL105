import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.distributions import Bernoulli

from model import Qnet, PolicyNet


class WorkerAgent(nn.Module):
    def __init__(self, config) -> None:
        super(WorkerAgent, self).__init__()
        self.q_net = Qnet(config.len_category, config.len_sub_category, config.len_industry, config.dim)
        self.target_q_net = Qnet(config.len_category, config.len_sub_category, config.len_industry, config.dim)
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.target_update = config.targe_update
        self.count = 0
    
    def take_action(self, state):
        worker_history, action_list = state
        project_index_list = list()


class MemoryQueue:
    def __init__(self):
        self.buffer = deque()
    
    def push(self, transitions):
        self.buffer.append(transitions)
        
    def sample(self):
        batch = list(self.buffer)
        return zip(*batch)
    
    def clear(self):
        self.buffer.clear()
        
    def __len__(self):
        return len(self.buffer)


class PolicyGradient:
    "PolicyGradient智能体对象"
    def __init__(self, memory, config):
        self.gamma = config.gamma
        self.memory = memory
        # 策略网络
        self.policy_net = PolicyNet()
        # 优化器     
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=config.lr)
        
    def sample_action(self, state):
        probs = self.policy_net(state)   
        m = Bernoulli(probs)
        action = m.sample()
        action = int(action.item())
        return action
    
    def update(self):
        state_pool, action_pool, reward_pool = self.memory.sample()
        state_pool, action_pool, reward_pool = list(state_pool), list(action_pool), list(reward_pool)
        # 对奖励进行修正，考虑未来，并加入衰减因子
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add
                
        # reawrd_mean = np.mean(reward_mean)
        # reward_std = np.std(reward_pool)  # 求奖励标准差
        # for i in range(len(reward_pool)):
        #     # 标准化奖励
        #     reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
        
        # 梯度下降
        self.optimizer.zero_grad()
        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = action_pool[i]
            reward = reward_pool[i]
            probs = self.policy_net(state)
            m = Bernoulli(probs)
            # 加权(reward)损失函数，加负号(将最大化问题转化为最小化问题)
            loss = -m.log_prob(action) * reward
            loss.backward()
        self.optimizer.step()
        self.memory.clear()
        
        
        
        