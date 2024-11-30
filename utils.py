import numpy as np
import torch


def expected_return(env, agent, seed, episodes=20):
    returns = np.zeros(episodes)
    with torch.no_grad():
        for i in range(episodes):
            s, _ = env.reset(seed=seed)
            done = False
            while not done:
                a = agent.sample_action(s)
                s_prime, r, terminated, truncated, info = env.step(a)
                s = s_prime
                returns[i] += r
                done = terminated or truncated
    return returns.mean(), returns.std(), returns
