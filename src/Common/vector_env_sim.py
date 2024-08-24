import asyncio

import gym
import numpy as np
from src.Common.common import background
from src.Common.sim import Sim


class MultiSims:
    def __init__(self, common, num_envs=1):
        self.num_envs = num_envs
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(*[self.make_env(common, i) for i in range(self.num_envs)])
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
    

    @background
    def make_env(self, common, i):        
        print(f"Making env idx:{i}")
        sim = Sim(common)
        return sim.env
    
    @background
    def reset_env(self, i):        
        return self.envs[i].reset()


    def step(self, action):
        return self.env.step(action)

    def close(self):
        [self.envs[i].close() for i in range(self.num_envs)]
        