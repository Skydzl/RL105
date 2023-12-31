#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:13:12 2023

@author: Skydzl
"""
import time
import yaml
import random
import pickle
import numpy as np

from tqdm import tqdm

from agent import PolicyGradientAgent
from utils import MemoryQueue
from env import WorkerEnv
from utils import plot_reward_curve, plot_loss_curve


class Config:
    "policynet的配置"
    def __init__(self):
        with open("./config/woker_policynet.yaml", "rb") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.env_name = config['env_name']
        self.alg = config['alg']
        self.epochs = config['epochs']
        self.ep_max_steps = config['ep_max_steps']
        self.update_fre = config['update_fre']
        self.gamma = config['gamma']
        self.len_category = config['len_category']
        self.len_sub_category = config['len_sub_category']
        self.len_industry = config['len_industry']
        self.dim = config['dim']
        self.learning_rate = config['learning_rate']
        self.num_projects = config['num_projects']
        
        self.worker_num = config["worker_num"]
        
def new_train(config, env, agent):
    start_time = time.time()
    print(f"环境名：{config.env_name}, 算法名：{config.alg}")
    print("开始训练智能体......")
    # 记录每个epoch的奖励
    
    # 每config.update_fre记录一次
    reward_list = []
    loss_list = []
    
    for epoch in range(config.epochs):
        iteration_return = 0
        step = 0
        env.reset()
        with tqdm(total=config.worker_num, desc='Episodes %d' % epoch) as worker_bar:
            for worker_iter in range(config.worker_num):
                # 进行一个回合
                state, done = env.get_obs()
                while not done:
                    action = agent.sample_action(state)
                    next_state, reward, done = env.step(action)
                    agent.memory.push((state, action, reward))
                    step += 1
                    state = next_state
                    iteration_return += reward
                    if step % config.update_fre == 0:
                        reward_list.append(iteration_return)
                        iteration_return = 0
                if len(agent.memory) != 0:
                    loss = agent.update()  
                    loss_list.append(loss)
                if (worker_iter + 1) % 10 == 0:
                    worker_bar.set_postfix({'episode': '%d' % (epoch + 1),
                                            'worker': '%d' % (worker_iter + 1),
                                            "loss_list": '%.3f' % np.mean(loss_list[-10: ]),
                                            "reward_list" : '%.3f' % np.mean(reward_list[-10: ])})
                env.worker_index += 1
                env.worker_list_pos = 0
                worker_bar.update(1)

    return reward_list, loss_list

def new_test(config, env, agent):
    reward_list = []
    reward_sum = 0
    accuracy_cnt = 0
    step = 0
    env.test_reset()
    for worker_iter in tqdm(range(config.worker_num)):
        state, done = env.get_obs()
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            step += 1
            if reward > 0:
                accuracy_cnt += 1
            state = next_state
            reward_list.append(reward)
            reward_sum += reward
        env.worker_index += 1
        env.worker_list_pos = 0
    accuracy = accuracy_cnt / step
    return reward_list, reward_sum, accuracy

def train(config, env, agent):
    start_time = time.time()
    loss_list = []
    print(f"环境名：{config.env_name}, 算法名：{config.alg}")
    print("开始训练智能体......")
    # 记录每个epoch的奖励
    rewards = []
    for epoch in range(config.epochs):
        state, done = env.reset()
        ep_reward = 0
        
        # 进行一个回合
        for _ in range(1, config.ep_max_steps):
            # 采样
            action = agent.sample_action(state)
            # 执行动作，获取下一个状态、奖励和结束状态
            next_state, reward, done = env.step(action)
            ep_reward += reward 
            # 如果回合结束，则奖励为0
            if done:
                reward = 0
            # 将采样的数据存起来
            agent.memory.push((state, action, reward))
            # 更新状态：当前状态等于下一个状态
            state = next_state
            if done:
                break
        if (epoch + 1) % 1 == 0:
            print(f"Epochs：{epoch + 1}/{config.epochs}, Reward:{ep_reward:.2f}")
        # 每采样几个回合就对智能体做一次更新
        if (epoch + 1) % config.update_fre == 0:
            loss = agent.update()  
            loss_list.append(loss)     
            print(f"train_loss: {loss:.2f}")          
        rewards.append(ep_reward)
    print('训练结束，用时：' + str(time.time() - start_time) + " s")
    return {'episodes': range(len(rewards)), 'rewards': rewards}, loss_list
    

def random_train(config, env, agent):
    start_time = time.time()
    print(f"环境名：{config.env_name}, 算法名：random")
    # 记录每个epoch的奖励
    rewards = []
    for epoch in range(config.epochs):
        state, done = env.reset()
        ep_reward = 0
        
        # 进行一个回合
        for _ in range(1, config.ep_max_steps):
            # 采样
            action = random.choice(state[1])
            # 执行动作，获取下一个状态、奖励和结束状态
            next_state, reward, done = env.step(action)
            '这里不需要乘gamma么？'
            ep_reward += reward 
            # 如果回合结束，则奖励为0
            if done:
                reward = 0
            # 将采样的数据存起来
            agent.memory.push((state, action, reward))
            # 更新状态：当前状态等于下一个状态
            state = next_state
            if done:
                break
        if (epoch + 1) % 1 == 0:
            print(f"Epochs：{epoch + 1}/{config.epochs}, Reward:{ep_reward:.2f}")   
        rewards.append(ep_reward)
    print('训练结束，用时：' + str(time.time() - start_time) + " s")
    return {'episodes': range(len(rewards)), 'rewards': rewards}


if __name__ == "__main__":
    config = Config()
    with open("./config/worker_dqn.yaml", "rb") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    memory = MemoryQueue()
    env = WorkerEnv(env_config)
    agent = PolicyGradientAgent(memory, config)

    train_policy_reward_list, train_policy_loss_list = new_train(config, env, agent)
    test_policy_reward_list, test_policy_reward_sum, test_policy_accuracy = new_test(config, env, agent)


    result_dict = {
        "train_policy_reward_list": train_policy_reward_list,
        "train_policy_loss_list": train_policy_loss_list,
        "test_policy_reward_list": test_policy_reward_list,
        "test_policy_reward_sum": test_policy_reward_sum,
        "test_policy_accuracy": test_policy_accuracy
    }

    with open("./result/Policy_Worker_result_dict.pickle", "wb") as fp:
        pickle.dump(result_dict, fp)
    # result, loss_list = train(config, env, agent)
    # random_result = random_train(config, env, agent)
    # result_list = result, loss_list, random_result
    # pickle.dump(result_list, open(
    #         "./result/Policy_Worker_result_list_his_{}_{}_{}.pickle"
    #         .format(str(config.ep_max_steps), str(config.epochs), str(config.update_fre)), "wb"))    
    # plot_reward_curve(result['rewards'], random_result['rewards'], "PolicyNet on Worker")
    # plot_loss_curve(loss_list, 'PolicyNet on Worker')