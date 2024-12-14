import asyncio

import gym
import numpy as np
from src.Common.common import Common, background
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from src.Common.wrapper import apply_wrappers


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
        loop.run_until_complete(looper)[0]

    def state_dict(self) -> dict:
        state_dict = {}
        if hasattr(self.env, "state_dict"):
            state_dict = self.env.state_dict()
        return state_dict

    def load_state_dict(self, state: dict):
        if hasattr(self.env, "load_state_dict"):
            self.env.load_state_dict(state)

    @background
    def make_env(self, common):
        print("Making env")
        env_name = common.config.get("ENV_NAME")
        env = gym_super_mario_bros.make(env_name)
        env.metadata["render.modes"] = ["human", "rgb_array"]
        env.metadata["apply_api_compatibility"] = True

        env = JoypadSpace(env, Common.RIGHT_RUN)
        env = apply_wrappers(env)
        env.reset()
        return env

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
