from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            # !WARNING trunc
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        # !WARNING trunc
        return next_state, total_reward, done, info

    def reset(self):
        return self.env.reset()


def apply_wrappers(env):
    env = SkipFrame(env, skip=4)
    # shape = 84
    env = ResizeObservation(env, shape=30)
    env = GrayScaleObservation(env, keep_dim=False)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env