#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:09:58 2023

@author: Skydzl
"""

# 测试函数
def test(arg_dict, env, agent):
    startTime = time.time()
    print("开始测试智能体......")
    print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    # 记录每个epoch的奖励
    rewards = []
    for epoch in range(arg_dict['test_eps']):
        state = env.reset()
        ep_reward = 0
        for _ in range(arg_dict['ep_max_steps']):
            # 画图
            if arg_dict['test_render']:
                env.render()
            action = agent.predict_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0
            state = next_state
            if done:
                break
        print(f"Epochs: {epoch + 1}/{arg_dict['test_eps']}，Reward: {ep_reward:.2f}")
        rewards.append(ep_reward)
    print("测试结束 , 用时: " + str(time.time() - startTime) + " s")
    env.close()
    return {'episodes': range(len(rewards)), 'rewards': rewards}


if __name__ == "__main__":
    
    