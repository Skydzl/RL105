import numpy as np
import random
import torch
import yaml

from tqdm import tqdm

from agent import WorkerAgent
from env import WorkerEnv
from utils import ReplayBuffer, moving_average, plot_return_curve

def train():
    with open("./config/worker_dqn.yaml", "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    replay_buffer = ReplayBuffer(config["buffer_size"])
    env = WorkerEnv()
    agent = WorkerAgent(config)
    return_list = []
    for i in range(10):
        with tqdm(total=int(config["num_episodes"] / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(config["num_episodes"] / 10)):
                episode_return = 0
                state, done = env.reset()
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    worker_history, action_list = state
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.cnt % config["update"] == 0:
                        transitions = replay_buffer.sample(config["batch_size"])
                        agent.update(transitions)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                    'episode':
                    '%d' % (config["num_episodes"] / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
                pbar.update(1)
    return return_list



if __name__ == "__main__":
    return_list = train()
    plot_return_curve(return_list, "Worker")