from collections import deque
from gymnasium import Wrapper


class MarkovWrapper(Wrapper):
    def __init__(self, env, delay=0, discretization=0, gamma=1):
        super().__init__(env)
        self.delay = delay
        self.queue = deque(maxlen=delay + 1)
        self.discretization = discretization
        self.gamma = gamma
        self.done = False

    def step(self, action):
        if not self.done:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.done = terminated or truncated
            if not self.done:
                for i in range(self.discretization):
                    obs, r, terminated, truncated, info = self.env.step(action)
                    # reward += (self.gamma ** (i + 1)) * r
                    reward += r
                    self.done = terminated or truncated
                    if self.done:
                        break
            self.queue.append((obs, reward, terminated, truncated, info))
        return self.queue.popleft()

    def reset(self, seed: int = None, **kwargs):
        self.queue.clear()
        self.done = False
        obs, info = self.env.reset(seed=seed, **kwargs)
        for _ in range(self.delay):
            self.queue.append((obs, 0, False, False, {}))
        return obs, info
