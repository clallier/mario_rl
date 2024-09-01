import math
import numpy as np
from gym import Wrapper
from gym.wrappers import (
    GrayScaleObservation,
    ResizeObservation,
    FrameStack,
    RecordEpisodeStatistics,
)


class RunningMeanStd:
    """Tracks the mean, variance and count of values.
    From: https://gymnasium.farama.org/_modules/gymnasium/wrappers/normalize
    Use float32!

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = np.float32(epsilon)

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0, dtype=np.float32)
        batch_var = np.var(x, axis=0, dtype=np.float32)
        batch_count = np.float32(x.shape[0])

        # Updates the mean, var and count using the previous mean, var, count and batch values
        mean, var, count = self.mean, self.var, self.count
        delta = batch_mean - mean
        tot_count = count + batch_count

        self.mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count


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


class NormalizeFrame(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = np.asarray(state, dtype=np.float32) / 255.0
        state = np.expand_dims(state, axis=0)
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        state = np.asarray(state, dtype=np.float32) / 255.0
        state = np.expand_dims(state, axis=0)
        return state


class StoreActions(Wrapper):
    """Store the list of actions taken"""

    def __init__(self, env):
        super().__init__(env)
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        state, reward, done, info = self.env.step(action)
        # append actions at the end of the episode
        if "episode" in info.keys():
            info["episode"]["a"] = self.actions
        return state, reward, done, info

    def reset(self):
        self.actions = []
        return self.env.reset()


class CustomReward(Wrapper):
    """Try to maximize the x-axis progress"""

    def __init__(self, env):
        super().__init__(env)
        self.previous_x_pos = 0
        self.previous_coins = 0
        self.previous_score = 0
        self.previous_status = "small"
        CustomReward.x_max = 2000
        CustomReward.y_min = 100
        CustomReward.y_max = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        x_pos = info["x_pos"].item()
        y_pos = info["y_pos"].item()
        v_x = 0.1 * abs(x_pos - self.previous_x_pos)
        flag_get = info["flag_get"]
        coins = info["coins"]
        score = info["score"] / 100
        status = (info["status"] != "small") * 1
        progress = x_pos / CustomReward.x_max

        reward = -0.5 + progress  # + 0.1 * v_x

        if progress > 1:
            reward += 1

        if flag_get:
            reward += 2000
        elif done and (y_pos < 75 or y_pos >= 252):
            # fall in a hole
            reward -= 1800
        elif done:
            # hit by a mob
            reward -= 2000

        # d_coins = coins - self.previous_coins
        # d_score = score - self.previous_score
        # if d_coins or d_score:
        #     reward += 100

        # reward += y * 5

        # d_status = status - self.previous_status

        self.previous_x_pos = x_pos
        self.previous_coins = coins
        self.previous_score = score
        self.previous_status = status
        if x_pos > CustomReward.x_max:
            CustomReward.x_max = x_pos
        if y_pos < CustomReward.y_min:
            CustomReward.y_min = y_pos
            print(f"### y_min: {CustomReward.y_min}")
        if y_pos > CustomReward.y_max:
            CustomReward.y_max = y_pos
            print(f"y_max: {CustomReward.y_max}")
        return state, reward, done, info

    def reset(self):
        self.previous_x_pos = 0
        return self.env.reset()

    def state_dict(self):
        return {"x_max": CustomReward.x_max}

    def load_state_dict(self, state: dict):
        CustomReward.x_max = state.get("x_max")


class NormalizeReward(Wrapper):
    """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
    From: https://gymnasium.farama.org/_modules/gymnasium/wrappers/normalize/#NormalizeReward

    Args:
        env (env): The environment to apply the wrapper
        epsilon (float): A stability parameter
        gamma (float): The discount factor that is used in the exponential moving average.
    """

    def __init__(self, env, gamma: float = 0.99, epsilon: float = 1e-8):
        super().__init__(env)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(1)
        self.gamma = np.float32(gamma)
        self.epsilon = np.float32(epsilon)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.returns = self.returns * self.gamma * (1 - done) + done
        info["reward"] = reward
        info["normalized_reward"] = self.normalize(reward)
        info["done"] = done
        return state, reward, done, info

    def normalize(self, reward):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return np.float32(reward) / np.sqrt(self.return_rms.var + self.epsilon)

    def reset(self):
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(1)
        return self.env.reset()

    def state_dict(self):
        return {
            "return_rms": self.return_rms,
            "returns": self.returns,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: dict):
        self.return_rms = state.get("return_rms")
        self.returns = state.get("returns")
        self.gamma = state.get("gamma")
        self.epsilon = state.get("epsilon")


def apply_wrappers(env):
    env = SkipFrame(env, skip=4)
    # shape = 30
    env = ResizeObservation(env, shape=30)
    env = GrayScaleObservation(env, keep_dim=False)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    env = NormalizeFrame(env)
    env = CustomReward(env)
    env = NormalizeReward(env)
    env = RecordEpisodeStatistics(env)
    env = StoreActions(env)
    return env
