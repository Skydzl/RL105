#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:13:12 2023

@author: Skydzl
"""
import time
import yaml

from agent import PolicyGradientAgent
from utils import MemoryQueue
from env import WorkerEnv


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
        

def train(config, env, agent):
    start_time = time.time()
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
        if (epoch + 1) % 10 == 0:
            print(f"Epochs：{epoch + 1}/{config.epochs}, Reward:{ep_reward:.2f}")
        # 每采样几个回合就对智能体做一次更新
        if (epoch + 1) % config.update_fre == 0:
            agent.update()                 
        rewards.append(ep_reward)
    print('训练结束，用时：' + str(time.time() - start_time) + " s")
    return {'episodes': range(len(rewards)), 'rewards': rewards}
    

if __name__ == "__main__":
    config = Config()
    memory = MemoryQueue()
    env = WorkerEnv()
    agent = PolicyGradientAgent(memory, config)
    train(config, env, agent)