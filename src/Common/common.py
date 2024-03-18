import pickle
import random
from datetime import datetime
from pathlib import Path, PosixPath

import numpy as np
import tomli
from configparser import ConfigParser

import torch
from torch.utils.tensorboard import SummaryWriter

import collections
from statistics import mean, stdev
import matplotlib.pyplot as plt

import shutil


def get_device():
    device = 'cpu'
    if torch.backends.mps.is_built():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    return device


def set_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(0)


class Common:
    # static values
    NUM_OF_EPISODES = 10_000
    # in episodes
    SAVE_FREQ = 1000
    ENV_NAME = 'SuperMarioBros-1-1-v3'
    RIGHT_RUN = [
        ['right', 'B'],
        ['right', 'A', 'B']
    ]
    device = get_device()
    debug = False
    set_seed()

    @staticmethod
    def load_config_file(config_file, parent=None):
        if not isinstance(config_file, PosixPath) and parent is None:
            parent = Path(__file__).parent
            config_file = Path(parent, config_file)
        print(f"Common.load_config_file: {config_file.resolve()}, exists: {config_file.exists()}")
        if config_file.suffix == 'toml':
            with open(config_file, "rb") as f:
                data = tomli.load(f)
        else:
            data = ConfigParser()
            data.read(str(config_file.resolve()), 'UTF-8')
            data = data._sections
        return data


class Logger:
    def __init__(self, start_tensorboard=True):
        if start_tensorboard is True:
            self.writer = SummaryWriter()
        else:
            self.writer = DummyLogger()

        self.log_dir = self.writer.log_dir
        self.checkpoint_dir = Path(self.log_dir, 'checkpoints/')
        self.actions_dir = Path(self.log_dir, 'actions/')
        Path(self.checkpoint_dir).mkdir(parents=True)
        Path(self.actions_dir).mkdir(parents=True)

    def add_scalar(self, name: str, value, step: int = -1):
        self.writer.add_scalar(name, value, step)

    def add_histogram(self, name: str, value, step: int = -1):
        self.writer.add_histogram(name, value, step)

    def add_hparams(self, hparams: dict, metrics: dict = None):
        if metrics is None:
            metrics = {}
        self.writer.add_hparams(hparams, metrics)

    def add_file(self, file_path):
        from_path = Path(file_path)
        to_path = Path(self.log_dir, from_path.name)
        shutil.copy(from_path, to_path)

    def add_figure(self, name, fig: plt.Figure):
        self.writer.add_figure(name, fig)

    def add_pickle(self, name, element):
        to_path = Path(self.log_dir, f'{name}.pkl')
        with open(to_path, 'wb') as f:
            pickle.dump(element, f, protocol=5)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class DummyLogger:
    def __init__(self):
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(Path.cwd(), "runs", date_str)

    def add_scalar(self, name: str, value, step: int):
        print(f"step: {step}, {name}: {value}")

    def add_histogram(self, name: str, value, step: int):
        pass

    def add_hparams(self, hparams: dict, metrics: dict):
        pass

    def add_figure(self, name, fig: plt.Figure):
        fig.savefig(Path(self.log_dir, f'{name}.svg'))

    def flush(self):
        pass

    def close(self):
        pass


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
        if isinstance(reward, dict) and 'normalized' in reward:
            self.rewards += reward['raw']
            self.logger.add_scalar("reward_raw", reward['raw'], episode)
            self.logger.add_scalar("reward_normalized", reward['normalized'], episode)
        else:
            self.rewards += reward
            self.logger.add_scalar("reward_raw", reward, episode)

    def end_of_episode(self, info, episode, save_actions=None):
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
