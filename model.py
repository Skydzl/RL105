import torch
import torch.nn as nn

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
    
    def forward(self, action):
        discrete_data, continuous_data = action
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
        self.empty_weight = nn.Parameter(torch.Tensor(dim))
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, 1)
        )
    
    def forward(self, worker_history, action):
        action_out = self.project_encoder(action)
        if len(worker_history) == 0:
            worker_out = self.empty_weight
        else:
            worker_out = torch.mean(torch.cat([self.project_encoder(state) for state in worker_history], dim=0).view(len(worker_history), -1), dim=0)
        mlp_input = torch.cat([action_out, worker_out], dim=-1)
        mlp_out = self.mlp(mlp_input)
        return mlp_out
