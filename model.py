import torch
import numpy as np
import torch.nn as nn
from utils import masked_softmax
import torch.nn.functional as F


from utils import model_init

class ProjectEncoder(nn.Module):
    def __init__(self, len_category, len_sub_catetory, len_industry, dim) -> None:
        super(ProjectEncoder, self).__init__()
        self.category_embedding = nn.Embedding(len_category, dim)
        self.sub_category_embedding = nn.Embedding(len_sub_catetory, dim)
        self.industry_embedding = nn.Embedding(len_industry, dim)
        self.fc = nn.Sequential(
            nn.Linear(dim * 3 + 3, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, project):
        discrete_data, continuous_data = project
        category, sub_category, industry = discrete_data
        average_score, client_feedback, total_awards_and_tips = continuous_data
        fc_input = torch.cat((
            self.category_embedding(category),
            self.sub_category_embedding(sub_category),
            self.industry_embedding(industry),
            average_score.view(-1),
            client_feedback.view(-1),
            total_awards_and_tips.view(-1),
        ), dim=-1)
        fc_out = self.fc(fc_input)
        return fc_out
    

class Qnet(nn.Module):
    def __init__(self, len_category, len_sub_category, len_industry, dim) -> None:
        super(Qnet, self).__init__()
        self.project_encoder = ProjectEncoder(len_category, len_sub_category, len_industry, dim)
        self.empty_weight = nn.Parameter(torch.Tensor(dim), requires_grad=True)
        self.empty_weight.data.uniform_(-1, 1)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, 1)
        )
        model_init(self)
    
    def forward(self, worker_history, action):
        action_out = self.project_encoder(action)
        if len(worker_history) == 0:
            worker_out = self.empty_weight
        else:
            worker_out = torch.mean(torch.cat([self.project_encoder(state) for state in worker_history], dim=0).view(len(worker_history), -1), dim=0)
        mlp_input = torch.cat([action_out, worker_out], dim=-1)
        mlp_out = self.mlp(mlp_input)
        return mlp_out


class PolicyNet(nn.Module):
    "策略网络（全连接网络）"
    def __init__(self, num_projects, len_category, len_sub_category, len_industry, dim):
        """ 初始化策略网络，为全连接网络
            len_category: 父标签向量的长度
            len_sub_category: 子标签向量长度
            len_industry: 子子标签向量长度
            dim: embedding_dim
        """
        super(PolicyNet, self).__init__()
        self.project_encoder = ProjectEncoder(len_category, len_sub_category, len_industry, dim)
        self.empty_weight = nn.Parameter(torch.Tensor(dim))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, num_projects),
        )

    def forward(self, worker_history, action_list):
        if len(worker_history) == 0:
            worker_out = self.empty_weight
        else:
            worker_out = torch.mean(torch.cat([self.project_encoder(state) for state in worker_history], dim=0).view(len(worker_history), -1), dim=0)
        mlp_input = worker_out
        mlp_out = self.mlp(mlp_input)   
        mask_index = torch.tensor([action[0] for action in action_list])
        return masked_softmax(mlp_out, mask_index)

class ProjectEncoderForAC(nn.Module):
    def __init__(self, len_category, len_sub_catetory, len_industry, dim, device):
        super(ProjectEncoderForAC, self).__init__()
        self.device = device
        self.category_embedding = nn.Embedding(len_category, dim)
        self.sub_category_embedding = nn.Embedding(len_sub_catetory, dim)
        self.industry_embedding = nn.Embedding(len_industry, dim)
        self.fc = nn.Sequential(
            nn.Linear(dim * 3 + 3, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, project):
        discrete_data, continuous_data = project
        category, sub_category, industry = discrete_data.to(self.device)
        average_score, client_feedback, total_awards_and_tips = continuous_data.to(self.device)
        fc_input = torch.cat((
            self.category_embedding(category),
            self.sub_category_embedding(sub_category),
            self.industry_embedding(industry),
            average_score.view(-1),
            client_feedback.view(-1),
            total_awards_and_tips.view(-1),
        ), dim=-1)
        fc_out = self.fc(fc_input)
        return fc_out

class ValueNetForAC(nn.Module):
    # ActorCritic算法的Value网络
    def __init__(self, dim):
        super(ValueNetForAC, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, 1),
        )
        model_init(self)

    def forward(self, worker_state):
        mlp_out = self.mlp(worker_state)
        return mlp_out
    
class PolicyNetForAC(nn.Module):
    # ActorCritic算法的Policy网络
    def __init__(self, num_projects, dim, device):
        """ 初始化策略网络，为全连接网络
            num_projects: 可选择的动作数量
            dim: embedding_dim
        """
        super(PolicyNetForAC, self).__init__()
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, num_projects),
        )

    def forward(self, worker_state, worker_action_list):
        mlp_out = self.mlp(worker_state)
        for idx, action_list in enumerate(worker_action_list):
            effective_idx = set(action[0] for action in action_list)
            mask_index = torch.tensor(list(set(np.arange(mlp_out.shape[-1])) - effective_idx)).to(self.device)
            mlp_out[idx][mask_index] = -1e6
        
        output = nn.functional.softmax(mlp_out, dim=-1)
        return output