import numpy as np
import torch
import torch.nn as nn

from model import Qnet

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
