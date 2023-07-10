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
            self.worker_time_list,
            self.worker_list,
            self.project_id2index_dict,
            self.project_index2id_dict,
            self.worker_id2index_dict,
            self.worker_index2id_dict,
            self.industry_dict,
            self.sub_category_dict,
            self.category_dict
        ) = data
        self.project_num = len(self.project_info)
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
        self.worker_pos = 0
        self.worker_answer_history_dict = {} # worker的回答历史
        self.project_answer_count = torch.zeros(self.project_num) # 当前项目回答的数量
        self.max_history_len = config["max_history_len"]
        self.true_history_len = config["true_history_len"]

    def reset(self):
        self.worker_pos = 0
        self.worker_answer_history_dict = {}
        self.project_answer_count = torch.zeros(self.project_num)

        while self.worker_pos < self.true_history_len:
            worker_id = self.worker_list[self.worker_pos]
            worker_time = self.worker_time_list[self.worker_pos]
            if worker_id not in self.worker_answer_history_dict:
                self.worker_answer_history_dict[worker_id] = []
            for project_id, p_info in self.project_info.items():
                project_index = self.project_id2index_dict[project_id]
                if p_info["start_date"] > worker_time or p_info["deadline"] < worker_time: # 时间不符合
                    continue
                if project_id in self.worker_answer_history_dict[worker_id]: # 已经回答过了
                    continue
                if self.project_answer_count[project_index] == p_info["entry_count"]: # project回答数量已满
                    continue
                if worker_id not in self.answer_info[project_id]:
                    continue
                self.worker_answer_history_dict[worker_id].append(project_id)
                self.project_answer_count[project_index] += 1
            self.worker_pos += 1
        
        done = False
        obs = None
        while not done:
            if self.worker_pos == len(self.worker_list):
                done = True
            if not done:
                obs = self._obs()
                worker_history, action_list = obs
                if len(action_list) > 0:
                    break
                self.worker_pos += 1
        return obs, done

    def step(self, action):
<<<<<<< HEAD
        project_index, _, __ = action
=======
        project_index, discrete, continuous = action
>>>>>>> 49ff65deef311e68d179c06519cbee4bb48555aa
        assert 0 <= project_index < self.project_num
        self.project_answer_count[project_index] += 1
        project_id = self.project_index2id_dict[project_index] # 记录该project的回答次数++
        worker_id = self.worker_list[self.worker_pos]
        
        if worker_id not in self.worker_answer_history_dict:
            self.worker_answer_history_dict[worker_id] = []
        reward = -1
        if worker_id in self.answer_info[project_id]:
            # 当前worker真实回答了该project
            self.worker_answer_history_dict[worker_id].append(project_id) # 记录当前worker已经回答了该project
            reward = 1
            if self.answer_info[project_id][worker_id]["finalist"] == True:
                # 当前worker的真实回答finalist
                reward = 3
                if self.answer_info[project_id][worker_id]["winner"] == True:
                    # 当前worker的真实回答是winner
                    reward = 5
        
        self.worker_pos += 1

        done = False
        obs = None
        while not done:
            if self.worker_pos == len(self.worker_list):
                done = True
            if not done:
                obs = self._obs()
                worker_history, action_list = obs
                if len(action_list) > 0:
                    break
                self.worker_pos += 1
        return obs, reward, done
    
    def _obs(self):
        worker_id = self.worker_list[self.worker_pos]
        worker_time = self.worker_time_list[self.worker_pos]
        action_list = list()
        if worker_id not in self.worker_answer_history_dict:
            self.worker_answer_history_dict[worker_id] = list()
        for project_id, p_info in self.project_info.items():
            project_index = self.project_id2index_dict[project_id]
            if p_info["start_date"] > worker_time or p_info["deadline"] < worker_time: # 时间不符合
                continue
            if project_id in self.worker_answer_history_dict[worker_id]: # 已经回答过了
                continue
            if self.project_answer_count[project_index] == p_info["entry_count"]: # project回答数量已满
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
        
