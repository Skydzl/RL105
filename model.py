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
    
    def forward(self, discrete_data, continuous_data):
        category, sub_category, industry = discrete_data
        average_score, client_feedback, total_awards_and_tips = continuous_data
        fc_input = torch.cat((
            self.category_embedding(category),
            self.sub_category_embedding(sub_category),
            self.industry_embedding(industry),
            average_score,
            client_feedback,
            total_awards_and_tips,
        ), dim=-1)
        fc_out = self.fc(fc_input)
        return fc_out
    

class Qnet(nn.Module):
    def __init__(self, len_category, len_sub_category, len_industry, dim) -> None:
        super(Qnet, self).__init__()
        self.project_encoder = ProjectEncoder(len_category, len_sub_category, len_industry, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, 1)
        )
    
    def forward(self, worker_state, action):
        action_out = self.project_encoder(action)
        worker_out = torch.mean([self.project_encoder(state) for state in worker_state], dim=-2)
        mlp_input = torch.cat([action_out, worker_out], dim=-1)
        mlp_out = self.mlp(mlp_input)
        return mlp_out


class PolicyNet(nn.Module):
    "策略网络（全连接网络）"
    def __init__(self, len_category, len_sub_category, len_industry, dim):
        """ 初始化策略网络，为全连接网络
            input_dim: 输入的特征数即环境的状态维度
            output_dim: 输出的动作维度
        """
        super(PolicyNet, self).__init__()
        self.project_encoder = ProjectEncoder(len_category, len_sub_category, len_industry, dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
    