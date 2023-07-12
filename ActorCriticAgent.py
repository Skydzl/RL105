import numpy as np
import time
import random
import torch
import torch.nn as nn
from collections import deque
from torch.distributions import Categorical

from model import ValueNetForAC, PolicyNetForAC, ProjectEncoderForAC
import torch.nn.functional as F
import os

class ActorCriticRandomAgent(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
    
    def take_action(self, state, mode="train"):
        worker_history, action_list = state
        action = random.choice(action_list)
        return action
    
    def update(self, transitions):
        return 0.0, 0.0

class ActorCriticAgent:
    def __init__(self, config):
        self.create_time = int(time.time())
        if (torch.cuda.is_available()) and (config["device"] == "gpu"):
            self.device = torch.device("cuda")
            os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
        else:
            self.device = torch.device("cpu")
        # 获得worker_history的embedding
        self.project_encoder = ProjectEncoderForAC(config["len_category"], config["len_sub_category"], config["len_industry"], config["dim"], self.device).to(self.device)
        # 如果历史为空，则用self.empty_weight表示worker_history
        self.empty_weight = nn.Parameter(torch.Tensor(config["dim"]), requires_grad=True).to(self.device)
        self.empty_weight.data.uniform_(-1, 1)

        # 策略网络
        self.actor = PolicyNetForAC(config["num_projects"], config["dim"], self.device).to(self.device)
        # 价值网络
        self.critic = ValueNetForAC(config["dim"]).to(self.device)

        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config["actor_lr"])
        # 价值网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])
        
        self.gamma = config["gamma"]
        self.config = config
    
    def worker_history_to_embedding(self, worker_history):
        if len(worker_history) == 0:
            worker_state = self.empty_weight
        else:
            worker_state = torch.mean(torch.cat([self.project_encoder(state) for state in worker_history], dim=0).view(len(worker_history), -1), dim=0)
        return worker_state.reshape(1, -1)

    def take_action(self, state):
        worker_history, action_list = state
        worker_state = self.worker_history_to_embedding(worker_history)
        probs = self.actor(worker_state, [action_list]) # 应该是mask掉后的，只有候选project的概率分布 
        m = Categorical(probs)
        action_index = m.sample()
        for item in action_list:
            if item[0] == action_index.item():
                action = item
        return action
        

    def update(self, transition_dict):
        states = transition_dict["states"]
        states_work_history = list(map(lambda x:x[0], states))
        states_action_list = list(map(lambda x:x[1], states))

        actions = transition_dict["actions"]
        action_idx = torch.tensor([action[0] for action in actions]).to(self.device).reshape(-1,1)
        rewards = transition_dict["rewards"]

        next_states = transition_dict["next_states"]
        next_states_work_history = list(map(lambda x:x[0], next_states))
        next_states_action_list = list(map(lambda x:x[1], next_states))

        dones = transition_dict["dones"]

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        worker_state = torch.concat([self.worker_history_to_embedding(worker_history) for worker_history in states_work_history])
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_worker_state = torch.concat([self.worker_history_to_embedding(worker_history) for worker_history in next_states_work_history])

    #     # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_worker_state)
        td_delta = td_target - self.critic(worker_state)  # 时序差分误差
        log_probs = torch.log(self.actor(worker_state, states_action_list).gather(1, action_idx))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(worker_state), td_target.detach()))
        actor_loss.backward(retain_graph=True)  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

        return actor_loss.item(), critic_loss.item()
    
    def save_model(self, mode):
        actor = self.actor.state_dict()
        critic = self.critic.state_dict()
        project_encoder = self.project_encoder.state_dict()

        torch.save({"actor" :actor,
                    "critic": critic,
                    "project_encoder": project_encoder}, "./model/{}/ActorCrict-{}".format(mode, self.create_time))