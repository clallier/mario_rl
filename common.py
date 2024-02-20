from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import collections
from statistics import mean, stdev

# static values
def get_device():
    device = 'cpu'
    if torch.backends.mps.is_built():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    return device


class Common:
    NUM_OF_EPISODES = 10_000
    ENV_NAME = 'SuperMarioBros-1-1-v3'
    RIGHT_RUN = [
        ['right', 'B'],
        ['right', 'A', 'B']
    ]

    device = get_device()

    # in episodes
    SAVE_FREQ = 1000
    # how often to sync target network with online network
    sync_network_rate = 1000

    # Hyperparameters
    learning_rate = 0.01
    learning_rate_start_factor = 1.
    learning_rate_end_factor = 0.00001
    learning_rate_decay = 0.999
    gamma = 0.99  # discount factor
    epsilon_init = 1.0  # exploration rate
    epsilon_decay = 0.99999965
    epsilon_min = 0.1
    batch_size = 32

class Logger:
    def __init__(self, start_logger=True):
        if start_logger is False:
            return
        self.writer = SummaryWriter()
        self.log_dir = self.writer.log_dir

        self.checkpoint_dir = Path(self.log_dir, 'checkpoints')
        self.actions_dir = Path(self.log_dir, 'actions')
        Path(self.checkpoint_dir).mkdir()
        Path(self.actions_dir).mkdir()

    def add_scalar(self, name: str, value, episode: int, step: int = -1):
        if self.writer is None:
            print(f"Episode {episode}, step: {step}, {name}: {value}")
        else:
            if step == -1:
                step = episode
            self.writer.add_scalar(name, value, step)

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()


class Tracker:
    def __init__(self, logger: Logger):
        self.logger = logger
        # accumulated rewards for the current episode 
        self.rewards = 0
        # current episode actions (for debugging)
        self.actions = []
        # best accumulated reward (for one episode) so far
        self.best_reward = 0
        # deque for the last 10 rewards
        self.d = collections.deque(maxlen=10)
        # Nmber of times the flag was reached
        self.flag_get_sum = 0

    def init_reward(self):
        self.rewards = 0

    def store_action(self, action, reward, episode):
        self.actions.append(action)
        if isinstance(reward, dict) and 'raw' in reward:
            self.rewards += reward['raw']
            self.logger.add_scalar("reward_raw", reward['raw'], episode)
            self.logger.add_scalar("reward_normalized", reward['normalized'], episode)
        else:
            self.rewards += reward
            self.logger.add_scalar("reward_raw", reward, episode)

    def end_of_episode(self, info, episode, save_actions = None):
        get_flag = info.get('flag_get', False) if info is not None else False
        self.d.append(self.rewards)
        self.flag_get_sum += get_flag
        avg_10 = mean(self.d) if len(self.d) > 2 else -1
        std_10 = stdev(self.d) if len(self.d) > 2 else -1
        if self.rewards > self.best_reward:
            self.best_reward = self.rewards
            if save_actions is None:
                print(f'"WARNING tracker.end_of_episode: save_actions is None, skipping save_actions"')
            else:
                save_actions(self.actions, episode, self.rewards)
            
        self.logger.add_scalar("rewards", self.rewards, episode)
        self.logger.add_scalar("avg_10", avg_10, episode)
        self.logger.add_scalar("std_10", std_10, episode)
        self.logger.add_scalar("flag_get_sum", self.flag_get_sum, episode)

        print(f'Episode {episode}, rewards: {self.rewards:.1f}, best so far: {self.best_reward:.1f}, '
                f'mean_10: {avg_10:.1f}, std_10: {std_10:.1f}')
