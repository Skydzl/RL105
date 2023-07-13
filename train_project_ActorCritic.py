import numpy as np
import pickle
import random
import torch
import yaml

from tqdm import tqdm

from ActorCriticAgent import ActorCriticAgent, ActorCriticRandomAgent
from env import WorkerEnv
from utils import ReplayBuffer, moving_average, plot_reward_curve, plot_loss_curve

def train(config, agent, env):
    episode_reward_list = []
    episode_actor_loss_list = []
    episode_critic_loss_list = []
    reward_episode_list = []
    for episode in range(config["num_episodes"]):
        env.reset()
        worker_reward_list = []
        worker_actor_loss_list = []
        worker_critic_loss_list = []
        reward_iterator_list = []
        with tqdm(total=config["worker_num"], desc='Episodes %d' % episode) as worker_iter:
            for i_worker in range(config["worker_num"]):
                worker_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, done = env.get_obs()
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    worker_id = env.worker_index2id_dict[env.worker_index_hash[env.worker_index]]
                    reward *= env.worker_quality[worker_id]
                    if done == False:
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        reward_iterator_list.append(reward)
                    if len(reward_iterator_list) == config["update_fre"]:
                        reward_episode_list.append(np.sum(reward_iterator_list))
                        reward_iterator_list = []
                    state = next_state
                    worker_return += reward
                worker_reward_list.append(worker_return)
                if len(transition_dict["states"]) != 0:
                    actor_loss, critic_loss = agent.update(transition_dict)
                    worker_actor_loss_list.append(actor_loss)
                    worker_critic_loss_list.append(critic_loss)

                if (i_worker + 1) % 10 == 0:
                    worker_iter.set_postfix({'episode': '%d' % (episode + 1),
                                            'worker': '%d' % (i_worker + 1),
                                            'worker_reward_list': '%.3f' % np.mean(worker_reward_list[-10:]),
                                            "worker_actor_loss_list" : '%.3f' % np.mean(worker_actor_loss_list[-10: ]),
                                            "worker_critic_loss_list" : '%.3f' % np.mean(worker_critic_loss_list[-10: ]),
                                            "reward_episode_list" : '%.3f' % np.mean(reward_episode_list[-10: ])})
                env.worker_index += 1
                env.worker_list_pos = 0
                worker_iter.update(1)
        episode_reward_list.append([np.mean(worker_reward_list), worker_reward_list])
        episode_actor_loss_list.append([np.mean(worker_actor_loss_list), worker_actor_loss_list])
        episode_critic_loss_list.append([np.mean(worker_critic_loss_list), worker_critic_loss_list])

    return reward_episode_list, episode_reward_list, episode_actor_loss_list, episode_critic_loss_list

def test(config, agent, env):
    reward_list = []
    reward_sum = 0
    accuracy_cnt = 0
    step = 0
    env.test_reset()
    for worker_iter in tqdm(range(config["worker_num"])):
        state, done = env.get_obs()
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            worker_id = env.worker_index2id_dict[env.worker_index_hash[env.worker_index]]
            reward *= env.worker_quality[worker_id]
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
    with open("./config/worker_ActorCritic.yaml", "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    agent = ActorCriticAgent(config)
    env = WorkerEnv(config)
    
    reward_episode_list, train_reward_list, train_actor_loss_list, train_critic_loss_list = train(config, agent, env)
    test_reward_list, test_reward_sum, accuracy = test(config, agent, env)

    result_dict = {
        "reward_episode_list": reward_episode_list,
        "train_reward_list": train_reward_list,
        "train_actor_loss_list":train_actor_loss_list,
        "train_critic_loss_list":train_critic_loss_list,
        "test_reward_list": test_reward_list,
        "test_reward_sum": test_reward_sum,
        "accuracy": accuracy
    }

    agent.save_model("project")
    with open("./result/project/ActorCrict-Project-{}.pickle".format(agent.create_time), "wb") as fp:
        pickle.dump(result_dict, fp)

    
    random_agent = ActorCriticRandomAgent(config)

    reward_episode_list, train_random_reward_list, _, _ = train(config, random_agent, env)
    test_random_reward_list, test_random_reward_sum, accuracy = test(config, random_agent, env)

    result_dict = {
        "reward_episode_list": reward_episode_list,
        "train_reward_list": train_random_reward_list,
        "test_reward_list": test_random_reward_list,
        "test_reward_sum": test_random_reward_sum,
        "accuracy": accuracy
    }

    with open("./result/project/ActorCrict-Project-random.pickle", "wb") as fp:
        pickle.dump(result_dict, fp)