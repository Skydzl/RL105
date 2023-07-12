import numpy as np
import pickle
import random
import torch
import yaml

from tqdm import tqdm

from agent import WorkerDQNAgent, WorkerRandAgent
from env import WorkerEnv
from utils import ReplayBuffer, moving_average, plot_reward_curve, plot_loss_curve

def train(config, agent, env):
    replay_buffer = ReplayBuffer(config["buffer_size"])
    
    reward_list = []
    loss_list = []
    # with tqdm(total=int(config["num_episodes"]), desc='Iteration %d' % i) as pbar:
    for episode in range(config["num_episodes"]):
        iteration_return = 0
        env.reset()
        for worker_iter in tqdm(range(config["worker_num"])):
        # while env.worker_index < config["worker_num"]:
            state, done = env.get_obs()
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                iteration_return += reward
                if replay_buffer.cnt % config["update_frequency"] == 0:
                    transitions = replay_buffer.sample(config["batch_size"])
                    loss = agent.update(transitions)
                    loss_list.append(loss)
                    reward_list.append(iteration_return)
                    iteration_return = 0
            env.worker_index += 1
            env.worker_list_pos = 0

    return reward_list, loss_list


def test(config, agent, env):
    reward_list = []
    reward_sum = 0
    accuracy_cnt = 0
    step = 0
    env.test_reset()
    for worker_iter in tqdm(range(config["worker_num"])):
        state, done = env.get_obs()
        while not done:
            action = agent.take_action(state, "test")
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

# def random_train():
#     with open("./config/worker_dqn.yaml", "rb") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     random.seed(config["seed"])
#     np.random.seed(config["seed"])
#     torch.manual_seed(config["seed"])
#     replay_buffer = ReplayBuffer(config["buffer_size"])
#     env = WorkerEnv(config)
#     reward_list = []
#     for episode in range(config["num_episodes"]):
#         iteration_return = 0
#         state, done = env.reset()
#         # while not done:
#         for i in tqdm(range(len(env.worker_list) - config["true_history_len"])):
#             worker_history, action_list = state
#             action = random.choice(action_list)
#             next_state, reward, done = env.step(action)
#             replay_buffer.add(state, action, reward, next_state, done)
#             state = next_state
#             iteration_return += reward
#             if replay_buffer.cnt % config["update_frequency"] == 0:
#                 # transitions = replay_buffer.sample(config["batch_size"])
#                 # agent.update(transitions)
#                 reward_list.append(iteration_return)
#                 iteration_return = 0
#             # if replay_buffer.cnt == 10000:
#             #     break
#             if done:
#                 break
#     return reward_list

# def train():
#     with open("./config/worker_dqn.yaml", "rb") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     random.seed(config["seed"])
#     np.random.seed(config["seed"])
#     torch.manual_seed(config["seed"])
#     replay_buffer = ReplayBuffer(config["buffer_size"])
#     env = WorkerEnv()
#     agent = WorkerAgent(config)
#     reward_list = []
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
#                 reward_list.append(episode_return)
#                 if (i_episode + 1) % 10 == 0:
#                     pbar.set_postfix({
#                     'episode':
#                     '%d' % (config["num_episodes"] / 10 * i + i_episode + 1),
#                     'return':
#                     '%.3f' % np.mean(reward_list[-10:])
#                 })
#                 pbar.update(1)
#     return reward_list



if __name__ == "__main__":
    with open("./config/worker_dqn.yaml", "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    dqn_agent = WorkerDQNAgent(config)
    random_agent = WorkerRandAgent(config)
    env = WorkerEnv(config)



    train_dqn_reward_list, train_dqn_loss_list = train(config, dqn_agent, env)
    dqn_agent.save('./model/dqn_agent.pt')
    test_dqn_reward_list, test_dqn_reward_sum, test_dqn_accuracy = test(config, dqn_agent, env)

    train_random_reward_list, train_random_loss_list = train(config, random_agent, env)
    test_random_reward_list, test_random_reward_sum, test_random_accuracy = test(config, random_agent, env)

    result_dict = {
        "train_dqn_reward_list": train_dqn_reward_list,
        "train_dqn_loss_list": train_dqn_loss_list,
        "test_dqn_reward_list": test_dqn_reward_list,
        "test_dqn_reward_sum": test_dqn_reward_sum,
        "test_dqn_accuracy": test_dqn_accuracy,
        "train_random_reward_list": train_random_reward_list,
        "train_random_loss_list": train_random_loss_list,
        "test_random_reward_list": test_random_reward_list,
        "test_random_reward_sum": test_random_reward_sum,
        "test_random_accuracy": test_random_accuracy
    }

    with open("./result/NEW_DQN_RANDOM_Worker_result_dict.pickle", "wb") as fp:
        pickle.dump(result_dict, fp)
    # print(reward_list)
    # plot_reward_curve(reward_list, random_list, "DQN on Worker")
    # plot_loss_curve(loss_list, 'DQN on Worker')
