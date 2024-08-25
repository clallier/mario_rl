import asyncio

import gym
import numpy as np
from src.Common.common import background
from src.Common.sim import Sim


class AsyncMultiSims:
    def __init__(self, common, num_envs=1):
        self.num_envs = num_envs
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(
            *[self.make_env(common, i) for i in range(self.num_envs)]
        )
        self.envs = loop.run_until_complete(looper)

        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

        assert isinstance(self.single_action_space, gym.spaces.Discrete)
        print(f"single_observation_space.shape {self.single_observation_space.shape}")
        print(f"single_action_space.n {self.single_action_space.n}")

    def reset(self):
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(*[self.reset_env(i) for i in range(self.num_envs)])
        states = loop.run_until_complete(looper)
        states = np.asarray(states, dtype=np.float32)

        print("### envs reset, states: ", states.min(), states.max())
        from matplotlib import pyplot as plt

        state2 = states.reshape([self.num_envs, -1, 30])
        plt.imshow(state2[0])

        return states

    def step(self, actions):
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(
            *[self.step_env(i, actions[i]) for i in range(self.num_envs)]
        )
        feedbacks = loop.run_until_complete(looper)

        next_obs, rewards, dones, infos = [], [], [], []
        for f in feedbacks:
            next_obs.append(f[0])
            rewards.append(f[1])
            dones.append(f[2])
            infos.append(f[3])

        return (
            np.asarray(next_obs),
            np.asarray(rewards),
            np.asarray(dones),
            np.asarray(infos),
        )

    def close(self):
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(*[self.close_env(i) for i in range(self.num_envs)])
        loop.run_until_complete(looper)

    @background
    def make_env(self, common, i):
        print(f"Making env idx:{i}")
        sim = Sim(common)
        return sim.env

    @background
    def reset_env(self, i):
        return self.envs[i].reset()

    @background
    def step_env(self, i, action):
        next_obs, reward, done, info = self.envs[i].step(action)
        if done:
            self.envs[i].reset()
        return next_obs, reward, done, info

    @background
    def close_env(self, i):
        self.envs[i].close()
