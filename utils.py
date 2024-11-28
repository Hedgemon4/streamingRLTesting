import numpy as np
import torch


# def expected_return(env, agent, seed, episodes=20):
#     returns = np.zeros(episodes)
#     with torch.no_grad():
#         for i in range(episodes):
#             s, _ = env.reset(seed=seed)
#             done = False
#             while not done:
#                 a = agent.sample_action(s)
#                 s_prime, r, terminated, truncated, info = env.step(a)
#                 s = s_prime
#                 if terminated or truncated:
#                     returns[i] = info['episode']['r'][0]
#                     terminated, truncated = False, False
#                     done = True
#     return returns.mean(), returns.std(), returns

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



# def expected_return(env, weights, gamma, episodes=100):
#     G = np.zeros(episodes)
#     for e in range(episodes):
#         s, _ = env.reset(seed=e)
#         done = False
#         t = 0
#         while not done:
#             phi = get_phi(s)
#             Q = np.dot(phi, weights).ravel()
#             a = eps_greedy_action(Q, 0.0)
#             s_next, r, terminated, truncated, _ = env.step(a)
#             done = terminated or truncated
#             G[e] += gamma**t * r
#             s = s_next
#             t += 1
#     return G.mean()