import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from src.Common.common import Common
from src.Common.wrapper import apply_wrappers


class Sim:
    def __init__(self, common):
        env_name = common.config.get("ENV_NAME")
        self.env = gym_super_mario_bros.make(env_name)
        self.env.metadata["render.modes"] = ["human", "rgb_array"]
        self.env.metadata["apply_api_compatibility"] = True

        self.env = JoypadSpace(self.env, Common.RIGHT_RUN)
        self.env = apply_wrappers(self.env)
        self.reset()

    def reset(self):
        state = self.env.reset()
        state, _, _, _ = self.env.step(0)
        # print("### env reset, state: ", state.min(), state.max())
        return state

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
