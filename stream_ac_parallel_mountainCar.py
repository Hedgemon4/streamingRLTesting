import csv
import multiprocessing
import os, pickle, argparse
from itertools import product
from multiprocessing import Pool

import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Categorical
from optim import ObGD as Optimizer
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init
from utils import expected_return
from wrapper import MarkovWrapper


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        pref = self.fc_pi(x)
        return pref

class Critic(nn.Module):
    def __init__(self, n_obs=11, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.linear_layer  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class StreamAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=32, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.value_net = Critic(n_obs=n_obs, hidden_size=hidden_size)

        self.optimizer_policy = Optimizer(self.policy_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.value_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)


    def pi(self, x):
        preferences = self.policy_net(x)
        probs = F.softmax(preferences, dim=-1)
        return probs

    def v(self, x):
        return self.value_net(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        probs = self.pi(x)
        dist = Categorical(probs)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor(np.array(a)), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        probs = self.pi(s)
        dist = Categorical(probs)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

def main(seed, observation_delay=0, repeat_interval=0, debug=False):
    torch.manual_seed(seed); np.random.seed(seed)
    env_name = "MountainCar-v0"

    lr = 1.0
    gamma = 0.99
    lamda = 0.8
    total_steps = 500_000
    entropy_coeff = 0.01
    kappa_policy = 3.0
    kappa_value = 2.0
    overshooting_info = 'store_true'

    num_training_timesteps = total_steps // (repeat_interval + 1)

    env = gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)
    env = MarkovWrapper(env, delay=observation_delay, discretization=repeat_interval, gamma=0.96)
    agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.n, lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    for t in range(1, num_training_timesteps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime, terminated or truncated, entropy_coeff, False)
        s = s_prime
        if terminated or truncated:
            if debug:
                print("Episodic Return: {}, Time Step {}".format(info['episode']['r'][0], t))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()
    env.close()

    # Final Eval
    eval_env = gym.make(env_name)
    eval_env = gym.wrappers.FlattenObservation(eval_env)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
    eval_env = NormalizeObservation(eval_env)
    eval_env = AddTimeInfo(eval_env)
    eval_env = MarkovWrapper(eval_env, delay=observation_delay, discretization=repeat_interval, gamma=0.96)
    mean_reward, std_reward, eval_returns = expected_return(eval_env, agent, seed, 10)
    eval_env.close()

    save_dir = f"constant_env_steps/mountainCar/data_stream_ac/interval_{repeat_interval}/delay_{observation_delay}/seed_{seed}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "training_data.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

    with open(os.path.join(save_dir, "results.csv"), "w") as file:
        writer = csv.writer(file)
        writer.writerow([mean_reward, std_reward])
        writer.writerow(eval_returns)

if __name__ == '__main__':
    print("Started script")
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=100_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--episode_steps', type=int, default=None)
    args = parser.parse_args()

    delays = [0, 1, 2, 3, 4, 5]
    intervals = [0, 4, 8, 12, 16]
    seeds = [np.random.randint(1, 10000001) for i in range(10)]

    combinations =list(product(seeds, delays, intervals))

    with Pool(multiprocessing.cpu_count() - 1) as p:
        p.starmap(main, combinations)
