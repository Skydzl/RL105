import numpy as np
import pickle
import random
import torch
import yaml

from tqdm import tqdm

from agent import WorkerAgent
from env import WorkerEnv
from utils import ReplayBuffer, moving_average, plot_reward_curve, plot_loss_curve

def train():
    with open("./config/worker_dqn.yaml", "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    replay_buffer = ReplayBuffer(config["buffer_size"])
    env = WorkerEnv(config)
    agent = WorkerAgent(config)
    return_list = []
    loss_list = []

    # with tqdm(total=int(config["num_episodes"]), desc='Iteration %d' % i) as pbar:
    for episode in range(config["num_episodes"]):
        iteration_return = 0
        state, done = env.reset()
        # while not done:
        for i in tqdm(range(len(env.worker_list) - config["true_history_len"])):
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            iteration_return += reward
            if replay_buffer.cnt % config["update_frequency"] == 0:
                transitions = replay_buffer.sample(config["batch_size"])
                loss = agent.update(transitions)
                loss_list.append(loss)
                return_list.append(iteration_return)
                iteration_return = 0
            # if replay_buffer.cnt == 10000:
            #     break
            if done:
                break
    return return_list, loss_list

def random_train():
    with open("./config/worker_dqn.yaml", "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    replay_buffer = ReplayBuffer(config["buffer_size"])
    env = WorkerEnv(config)
    return_list = []
    for episode in range(config["num_episodes"]):
        iteration_return = 0
        state, done = env.reset()
        # while not done:
        for i in tqdm(range(len(env.worker_list) - config["true_history_len"])):
            worker_history, action_list = state
            action = random.choice(action_list)
            next_state, reward, done = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            iteration_return += reward
            if replay_buffer.cnt % config["update_frequency"] == 0:
                # transitions = replay_buffer.sample(config["batch_size"])
                # agent.update(transitions)
                return_list.append(iteration_return)
                iteration_return = 0
            # if replay_buffer.cnt == 10000:
            #     break
            if done:
                break
    return return_list

# def train():
#     with open("./config/worker_dqn.yaml", "rb") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     random.seed(config["seed"])
#     np.random.seed(config["seed"])
#     torch.manual_seed(config["seed"])
#     replay_buffer = ReplayBuffer(config["buffer_size"])
#     env = WorkerEnv()
#     agent = WorkerAgent(config)
#     return_list = []
#     for i in range(10):
#         with tqdm(total=int(config["num_episodes"] / 10), desc='Iteration %d' % i) as pbar:
#             for i_episode in range(int(config["num_episodes"] / 10)):
#                 episode_return = 0
#                 state, done = env.reset()
#                 while not done:
#                     action = agent.take_action(state)
#                     next_state, reward, done = env.step(action)
#                     worker_history, action_list = state
#                     replay_buffer.add(state, action, reward, next_state, done)
#                     state = next_state
#                     episode_return += reward
#                     if replay_buffer.cnt % config["update"] == 0:
#                         transitions = replay_buffer.sample(config["batch_size"])
#                         agent.update(transitions)
#                 return_list.append(episode_return)
#                 if (i_episode + 1) % 10 == 0:
#                     pbar.set_postfix({
#                     'episode':
#                     '%d' % (config["num_episodes"] / 10 * i + i_episode + 1),
#                     'return':
#                     '%.3f' % np.mean(return_list[-10:])
#                 })
#                 pbar.update(1)
#     return return_list



if __name__ == "__main__":
    return_list, loss_list = train()
    random_list = random_train()
    result_list = return_list, random_list, loss_list
    with open("./result/DQN_Worker_result_list_his_190000.pickle", "wb") as fp:
        pickle.dump(result_list, fp)
    # print(return_list)
    plot_reward_curve(return_list, random_list, "DQN on Worker")
    plot_loss_curve(loss_list, 'DQN on Worker')
