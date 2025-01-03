from abc import ABC, abstractmethod
from pathlib import Path
import re
import numpy as np
import torch

from src.Common.common import Common, Logger, Tracker
from src.Common.run_from_actions import test_from_actions

class Trainer(ABC):
    def __init__(self, common: Common, algo: str):
        self.algo = algo.lower()
        self.common = common
        self.logger = Logger(common)
        self.tracker = Tracker(self.algo, self.logger)

        # config
        self.config_path = common.get_config_path(self.algo)
        self.version = common.get_version(self.config_path)
        self.config = common.load_config_file(self.config_path)
        self.logger.add_file(self.config_path)

        self.debug = common.debug
        self.use_save_states = common.config.get("use_save_states", True)
        self.save_states_freq = common.config.get("save_states_freq", 100)
        self.episode = 0
        self.num_episodes = self.config.get("NUM_OF_EPISODES")
        self.info_shape = common.config.get("info_shape", (6,))
        self.device = common.device

        self.sim = self.create_sim()
        self.agent = self.create_agent()
        self.init()
        self.load_state()

    def close(self):
        if self.sim:
            self.sim.close()
        if self.logger:
            self.logger.close()

    def to_tensor(self, arr: np.array, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=self.device)

    def info_to_tensor(self, info: np.array):
        arr = np.array(
            [
                info.get("x_pos", 0),
                info.get("y_pos", 80),
                info.get("flag_get", 0),
                info.get("coins", 0),
                info.get("score", 0),
                info.get("time", 400),
            ]
        )
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def create_sim(self):
        pass

    @abstractmethod
    def create_agent(self):
        pass

    def train(self):
        self.train_init()
        from_episode = self.episode + 1
        for episode in range(from_episode, self.num_episodes + 1):
            self.episode = episode
            print(f"Episode {episode} started")
            info = self.run_episode(episode)
            # end of episode
            self.end_of_episode(info, episode)
        self.close()

    def end_of_episode(self, info, episode):
        self.tracker.end_of_episode(self.agent, info, episode)
        self.logger.flush()
        if self.use_save_states and episode % self.save_states_freq == 0:
            path = self.save_state()
            actions = info.get("episode").get("a")
            test_from_actions(actions, self.common, path)

    @abstractmethod
    def train_init(self):
        pass

    @abstractmethod
    def run_episode(self, episode: int) -> dict:
        pass

    def save_state(self):
        can_overwrite = True
        path = Path(self.logger.checkpoint_dir, f"{self.algo}_{self.version}_last.pt")
        if not can_overwrite and path.exists():
            print(f"WARNING save_state: path already exists, skipping save: {path}")
            return
        self.save_complete_state(path)
        print("state saved!")
        return path

    @abstractmethod
    def save_complete_state(self, path: Path):
        pass

    def load_state(self):
        if not self.use_save_states:
            return
        path_dir = Path(self.logger.checkpoint_dir)
        ckpt_found = False
        p = path_dir.glob("*.pt")
        files = [x for x in p if x.is_file()]
        for file in files:
            match = re.match(r"(.+)_(.+)_last.pt", file.name)
            if match:
                algo, version = match.groups()
                if algo == self.algo and version == self.version:
                    print(f"ckpt found: {file.name}, algo: {algo}, version: {version}")
                    ckpt_found = True

        if ckpt_found:
            path = Path(
                self.logger.checkpoint_dir, f"{self.algo}_{self.version}_last.pt"
            )
            self.load_complete_state(path)
            print("state loaded!")

    @abstractmethod
    def load_complete_state(self, path: Path):
        pass
