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
    
    reward_list = []
    loss_list = []
    # with tqdm(total=int(config["num_episodes"]), desc='Iteration %d' % i) as pbar:
    for episode in range(config["num_episodes"]):
        replay_buffer = ReplayBuffer(config["buffer_size"])
        iteration_return = 0
        env.reset()
        with tqdm(total=config["worker_num"], desc='Episodes %d' % episode) as worker_bar:
            for worker_iter in range(config["worker_num"]):
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
                if (worker_iter + 1) % 10 == 0:
                    worker_bar.set_postfix({'episode': '%d' % (episode + 1),
                                            'worker': '%d' % (worker_iter + 1),
                                            "loss_list": '%.3f' % np.mean(loss_list[-10: ]),
                                            "reward_list" : '%.3f' % np.mean(reward_list[-10: ])})
                env.worker_index += 1
                env.worker_list_pos = 0
                worker_bar.update(1)
                

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

    dqn_result_dict = {
        "train_dqn_reward_list": train_dqn_reward_list,
        "train_dqn_loss_list": train_dqn_loss_list,
        "test_dqn_reward_list": test_dqn_reward_list,
        "test_dqn_reward_sum": test_dqn_reward_sum,
        "test_dqn_accuracy": test_dqn_accuracy
    }

    with open("./result/DQN_Worker_result_dict.pickle", "wb") as fp:
        pickle.dump(dqn_result_dict, fp)

    train_random_reward_list, train_random_loss_list = train(config, random_agent, env)
    test_random_reward_list, test_random_reward_sum, test_random_accuracy = test(config, random_agent, env)

    random_result_dict = {
        "train_random_reward_list": train_random_reward_list,
        "train_random_loss_list": train_random_loss_list,
        "test_random_reward_list": test_random_reward_list,
        "test_random_reward_sum": test_random_reward_sum,
        "test_random_accuracy": test_random_accuracy
    }

    with open("./result/RANDOM_Worker_result_dict.pickle", "wb") as fp:
        pickle.dump(random_result_dict, fp)
