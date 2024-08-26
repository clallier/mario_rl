import asyncio

import gym
import numpy as np
from src.Common.common import background
from src.Common.sim import Sim


class AsyncSingleSim:
    def __init__(self, common):
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(self.make_env(common))
        self.env = loop.run_until_complete(looper)[0]

        self.single_observation_space = self.env.observation_space
        self.single_action_space = self.env.action_space

        assert isinstance(self.single_action_space, gym.spaces.Discrete)
        print(f"single_observation_space.shape {self.single_observation_space.shape}")
        print(f"single_action_space.n {self.single_action_space.n}")

    def reset(self):
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(self.reset_env())
        state = loop.run_until_complete(looper)[0]
        state = np.asarray(state, dtype=np.float32)
        return state

    def step(self, action):
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(self.step_env(action))
        feedback = loop.run_until_complete(looper)[0]
        return feedback

    def render(self):
        self.env.render()

    def close(self):
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(self.close_env())
        feedback = loop.run_until_complete(looper)[0]
        return feedback

    @background
    def make_env(self, common):
        print("Making env")
        sim = Sim(common)
        return sim.env

    @background
    def reset_env(self):
        next_obs = self.env.reset()
        return np.expand_dims(next_obs, axis=0)

    @background
    def step_env(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if done:
            self.env.reset()
        return np.expand_dims(next_obs, axis=0), reward, done, info

    @background
    def close_env(self):
        return self.env.close()

    @background
    def render_env(self):
        # RuntimeError: rendering from python threads is not supported
        return self.env.render()
