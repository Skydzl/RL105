import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Qnet

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




