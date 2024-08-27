from abc import ABC, abstractmethod
from pathlib import Path
import re

from src.Common.common import Common, Logger


class Trainer(ABC):

    def __init__(self, common: Common, algo: str):
        self.common = common
        self.logger = Logger(common)

        # config
        self.algo = algo.lower()
        config_file_name = f"{self.algo}_config_file"
        config_path = Common.get_file(
            f"./config_files/{common.config.get(config_file_name)}"
        )

        self.version = self.get_version(config_path.stem)
        self.config = common.load_config_file(config_path)
        self.logger.add_file(config_path)

        self.debug = common.config.get("debug")
        self.save_freq = common.config.get("save_freq", 100)
        self.episode = -1
        self.num_episodes = self.config.get("NUM_OF_EPISODES")
        self.device = common.device

        self.sim = self.create_sim()
        self.agent = self.create_agent()
        self.init()
        self.load_state()

    def get_version(self, config_path_stem: str):
        match = re.match(r".+.v(.+)", config_path_stem)
        version = match.groups() if match else "0"
        return "v" + version

    def close(self):
        if self.sim:
            self.sim.close()
        if self.logger:
            self.logger.close()

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
        from_episode = self.episode + 1
        for episode in range(from_episode, self.num_episodes):
            self.episode = episode
            self.run_episode(episode + 1)
        self.close()

    def run_episode(self, episode: int):
        print(f"Episode {episode} started")
        self.run_single_episode(episode)
        if episode % self.save_freq == 0:
            self.save_state()

    @abstractmethod
    def run_single_episode(self, episode: int):
        pass

    def save_state(self):
        can_overwrite = True
        path = Path(self.logger.checkpoint_dir, f"{self.algo}_{self.version}_last.pt")
        if not can_overwrite and path.exists():
            print(f"WARNING save_state: path already exists, skipping save: {path}")
            return
        self.save_complete_state(path)

    @abstractmethod
    def save_complete_state(self, path: Path):
        pass

    def load_state(self):
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

    @abstractmethod
    def load_complete_state(self, path: Path):
        pass
