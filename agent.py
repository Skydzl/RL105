import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.distributions import Bernoulli

from model import Qnet, PolicyNet
import torch.nn.functional as F


class WorkerAgent(nn.Module):
    def __init__(self, config) -> None:
        super(WorkerAgent, self).__init__()
        self.q_net = Qnet(config["len_category"], config["len_sub_category"], config["len_industry"], config["dim"])
        self.target_q_net = Qnet(config["len_category"], config["len_sub_category"], config["len_industry"], config["dim"])
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=config["learning_rate"])
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.target_update = config["target_update"]
        self.count = 0
    
    def take_action(self, state):
        worker_history, action_list = state
        project_index_list = list()
        project_index_list = [project_index for project_index, discrete, continuous in action_list]
        if len(worker_history) == 0 or np.random.random() < self.epsilon:
            action = np.random.choice(project_index_list)
        else:
            max_q_value = 0.
            for project_index, discrete, continuous in action_list:
                q_value = self.q_net(worker_history, (discrete, continuous))
                if q_value > max_q_value:
                    action = project_index
                    max_q_value = q_value
        return action
    
    def update(self, transitions):
        loss_list = list()
        for state, action, reward, next_state, done in transitions:
            worker_history, action_list = state
            discrete, continuous = action
            next_worker_history, next_action_list = next_state

            q_values = self.q_net(worker_history, (discrete, continuous))
            max_next_q_values = max([self.target_q_net(next_worker_history, (next_discrete, next_continuous))
                                for next_project_index, next_discrete, next_continuous in next_action_list])
            q_target = reward + self.gamma * max_next_q_values * (1 - done)
            loss_list.append(F.mse_loss(q_values, q_target))
        dqn_loss = torch.mean(torch.cat(loss_list, dim=-1), dim=-1)
        self.opt.zero_grad()
        dqn_loss.backward()
        self.opt.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict()
            )
        self.count += 1


class PolicyGradientAgent:
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
        
        def save_model(self, path):
            torch.save(self.policy_net, path+'policy_net.pth')
            
        def load_model(self, path):
            self.policy_net = torch.load(path+'policy_net.pth')
