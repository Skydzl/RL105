import copy
import pickle 
import torch
import random
from utils import to_onehot

class WorkerEnv(object):
    def __init__(self, config) -> None:
        with open('./data/env_data.pickle', 'rb') as f:
            data = pickle.load(f)
        (
            self.worker_quality,
            self.project_info,
            self.answer_info,
            _, # self.worker_time_list, 使用train_data的time_list作为worker_time_list
            __, # self.worker_list, 使用train_data的worker_list作为worker_list
            self.project_id2index_dict,
            self.project_index2id_dict,
            self.worker_id2index_dict,
            self.worker_index2id_dict,
            self.industry_dict,
            self.sub_category_dict,
            self.category_dict
        ) = data
        with open('./data/split_data.pickle', 'rb') as f:
            split_data = pickle.load(f)
        (
            self.prior_worker_history_dict, # {worker_id: [(time, project_id)]}
            self.train_worker_history_dict, # {worker_id: [(time, project_id)]}
            self.valid_worker_history_dict, # {worker_id: [(time, project_id)]}
            self.test_worker_history_dict   # {worker_id: [(time, project_id)]}
        ) = split_data

        self.project_num = len(self.project_info)
        self.worker_num = len(self.worker_quality)
        self.project_discrete_vector = torch.zeros((self.project_num, 3), dtype=torch.int64)
        self.project_continuous_vector = torch.zeros((self.project_num, 3), dtype=torch.float)
        for p_id, p_info in self.project_info.items():
            project_index = self.project_id2index_dict[p_id]
            self.project_discrete_vector[project_index][0] = self.category_dict[p_info["category"]]
            self.project_discrete_vector[project_index][1] = self.sub_category_dict[p_info["sub_category"]]
            self.project_discrete_vector[project_index][2] = self.industry_dict[p_info["industry"]]
            self.project_continuous_vector[project_index][0] = p_info["average_score"]
            self.project_continuous_vector[project_index][1] = p_info["client_feedback"]
            self.project_continuous_vector[project_index][2] = p_info["total_awards_and_tips"]
        self.worker_index = 0 # 当前是哪个worker
        self.worker_list_pos = 0 # 当前worker的状态位置
        self.worker_answer_history_dict = {} # worker的回答历史
        self.project_answer_count = torch.zeros(self.project_num) # 当前项目回答的数量
        self.max_history_len = config["max_history_len"]

    def reset(self): 
        # 训练阶段每回合的初始化
        self.worker_index = 0
        self.worker_list_pos = 0
        self.worker_answer_history_dict = {}


        for worker_id, history_list in self.prior_worker_history_dict.items():
            self.worker_answer_history_dict[worker_id] = list()
            for time, project_id in history_list:
                self.worker_answer_history_dict[worker_id].append(project_id)
                
        obs, done = self.get_obs()
        return obs, done

    # TODO: 还没修改完
    def test_reset(self, mode="test"):
        # 测试阶段的初始化
        if mode == "test":
            self.worker_time_list, self.worker_list, _ = zip(*sorted(self.test_data))
        elif mode == "valid":
            self.worker_time_list, self.worker_list, _ = zip(*sorted(self.valid_data))
        self.worker_pos = 0
        self.worker_answer_history_dict = {}
        self.project_answer_count = torch.zeros(self.project_num)

        for time, worker_id, project_id in self.prior_data:
            project_index = self.project_id2index_dict[project_id]
            self.project_answer_count[project_index] += 1
            if worker_id not in self.worker_answer_history_dict:
                self.worker_answer_history_dict[worker_id] = list()
            self.worker_answer_history_dict[worker_id].append(project_id)
        
        for time, worker_id, project_id in self.train_data:
            project_index = self.project_id2index_dict[project_id]
            self.project_answer_count[project_index] += 1
            if worker_id not in self.worker_answer_history_dict:
                self.worker_answer_history_dict[worker_id] = list()
            self.worker_answer_history_dict[worker_id].append(project_id)
        
        if mode == "test":
            for time, worker_id, project_id in self.valid_data:
                project_index = self.project_id2index_dict[project_id]
                self.project_answer_count[project_index] += 1
                if worker_id not in self.worker_answer_history_dict:
                    self.worker_answer_history_dict[worker_id] = list()
                self.worker_answer_history_dict[worker_id].append(project_id)
        
        obs, done = self.get_obs()
        return obs, done

    def step(self, action):
        project_index, discrete, continuous = action
        assert 0 <= project_index < self.project_num
        # self.project_answer_count[project_index] += 1  # 记录该project的回答次数++
        project_id = self.project_index2id_dict[project_index]
        worker_id = self.worker_index2id_dict[self.worker_index]
        
        reward = 0
        self.worker_answer_history_dict[worker_id].append(project_id) # 记录当前worker已经回答了该project
        if worker_id in self.answer_info[project_id]:
            # 当前worker真实回答了该project
            reward = self.answer_info[project_id][worker_id]["answer_cnt"] # 奖励值为 (回答这个问题的次数)
            if self.answer_info[project_id][worker_id]["finalist"] == True:
                # 当前worker的真实回答是finalist
                reward += 5
                if self.answer_info[project_id][worker_id]["winner"] == True:
                    # 当前worker的真实回答是winner
                    reward += 10
        
        self.worker_list_pos += 1

        obs, done = self.get_obs()
        return obs, reward, done
    
    def get_obs(self):
        done = False
        obs = None
        while not done:
            worker_id = self.worker_index2id_dict[self.worker_index]
            if self.worker_list_pos == len(self.train_worker_history_dict[worker_id]):
                done = True
            if not done:
                obs = self._obs()
                worker_history, action_list = obs
                if len(action_list) > 0:
                    break
                self.worker_list_pos += 1
        return obs, done

    def _obs(self):
        worker_id = self.worker_index2id_dict[self.worker_index]
        worker_time = self.train_worker_history_dict[self.worker_id][self.worker_list_pos][0]
        action_list = list()
        for project_id, p_info in self.project_info.items():
            project_index = self.project_id2index_dict[project_id]
            if p_info["start_date"] > worker_time or p_info["deadline"] < worker_time: # 时间不符合
                continue
            if project_id in self.worker_answer_history_dict[worker_id]: # 已经回答过了
                continue
            action = (
                project_index,
                self.project_discrete_vector[project_index],
                self.project_continuous_vector[project_index],
            )
            action_list.append(action)
        worker_history = list()
        if len(self.worker_answer_history_dict[worker_id]) > self.max_history_len:
            for p_id in self.worker_answer_history_dict[worker_id][-self.max_history_len:]:
                p_index = self.project_id2index_dict[p_id]
                worker_history.append((self.project_discrete_vector[p_index], self.project_continuous_vector[p_index]))
        else:
            for p_id in self.worker_answer_history_dict[worker_id]:
                p_index = self.project_id2index_dict[p_id]
                worker_history.append((self.project_discrete_vector[p_index], self.project_continuous_vector[p_index]))
        return worker_history, action_list


if __name__ == "__main__":
    wkenv = WorkerEnv()
        
