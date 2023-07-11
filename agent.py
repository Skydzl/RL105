import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque
from torch.distributions import Categorical

from model import Qnet, PolicyNet
import torch.nn.functional as F


class WorkerRandAgent(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
    
    def take_action(self, state, mode="train"):
        worker_history, action_list = state
        action = random.choice(action_list)
        return action
    
    def update(self, transitions):
        return 0.

class WorkerDQNAgent(nn.Module):
    def __init__(self, config) -> None:
        super(WorkerDQNAgent, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and config["device"] == "gpu" else "cpu")
        self.q_net = Qnet(config["len_category"], config["len_sub_category"], config["len_industry"], config["dim"]).to(self.device)
        self.target_q_net = Qnet(config["len_category"], config["len_sub_category"], config["len_industry"], config["dim"]).to(self.device)
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=config["learning_rate"])
        self.count = 0
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.target_update = config["target_update"]
        
    @torch.no_grad()
    def take_action(self, state, mode="train"):
        worker_history, action_list = state
        # project_index_list = [project_index for project_index, discrete, continuous in action_list]
        if mode == "train" and np.random.random() < self.epsilon + 1.0 / (self.count / 10 + 1.5):
            res_action = random.choice(action_list)
        else:
            res_action = None
            max_q_value = 0.
            for action in action_list:
                project_index, discrete, continuous = action
                discrete = discrete.to(self.device)
                continuous = continuous.to(self.device)
                for i in range(len(worker_history)):
                    worker_history[i] = tuple(x.to(self.device) for x in worker_history[i])
                q_value = self.q_net(worker_history, (discrete, continuous)).item()
                if q_value > max_q_value:
                    res_action = action
                    max_q_value = q_value
            if res_action is None:
                res_action = random.choice(action_list)
        return res_action
    
    def update(self, transitions):
        loss_list = list()
        for state, action, reward, next_state, done in transitions:
            worker_history, action_list = state
            project_index, discrete, continuous = action

            discrete = discrete.to(self.device)
            continuous = continuous.to(self.device)
            for i in range(len(worker_history)):
                worker_history[i] = tuple(x.to(self.device) for x in worker_history[i])

            q_values = self.q_net(worker_history, (discrete, continuous))
            if done:
                q_target = torch.Tensor([reward])
            else:
                next_worker_history, next_action_list = next_state
                for i in range(len(next_worker_history)):
                    next_worker_history[i] = tuple(x.to(self.device) for x in next_worker_history[i])

                max_next_q_values = max([self.target_q_net(next_worker_history, (next_discrete.to(self.device), next_continuous.to(self.device)))
                                    for next_project_index, next_discrete, next_continuous in next_action_list])
                q_target = reward + self.gamma * max_next_q_values
            # print('q_values:', q_values)
            # print('max_next_q_values:', max_next_q_values)
            # print('q_target:', q_target)
            loss_list.append(F.mse_loss(q_values, q_target.detach().to(self.device)).view(-1))
        dqn_loss = torch.mean(torch.cat(loss_list, dim=-1), dim=-1)
        self.opt.zero_grad()
        dqn_loss.backward()
        self.opt.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict()
            )
        self.count += 1
        return dqn_loss.cpu().item()


class PolicyGradientAgent:
    "PolicyGradient智能体对象"
    def __init__(self, memory, config):
        self.gamma = config.gamma
        self.memory = memory
        # 策略网络
        self.policy_net = PolicyNet(config.num_projects, config.len_category, config.len_sub_category
                                    ,config.len_industry, config.dim)
        # 优化器     
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=config.learning_rate)
        
    def sample_action(self, state):
        worker_history, action_list = state 
        probs = self.policy_net(worker_history, action_list) # 应该是mask掉后的，只有候选project的概率分布 
        m = Categorical(probs)
        action_index = m.sample()
        action = [action for action in action_list if action[0] == int(action_index.item())][0]
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
        loss_list = []
        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = torch.tensor(action_pool[i][0])
            reward = reward_pool[i]
            probs = self.policy_net(state[0], state[1])
            m = Categorical(probs)
            # 加权(reward)损失函数，加负号(将最大化问题转化为最小化问题)
            loss = -m.log_prob(action) * (reward + 1e-6)
            loss.backward()
            loss_list.append(loss.detach())
        self.optimizer.step()
        self.memory.clear()
        return np.array(loss_list).mean()
        
        def save_model(self, path):
            torch.save(self.policy_net, path+'policy_net.pth')
            
        def load_model(self, path):
            self.policy_net = torch.load(path+'policy_net.pth')
