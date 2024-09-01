import asyncio
import collections
import math
import os
import pickle
import random
import shutil
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path, PosixPath
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import tomli
import torch
import wandb
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


class Common:
    RIGHT_RUN = [["right"], ["right", "A"], ["right", "A", "B"]]
    # RIGHT_RUN = [["right", "B"], ["right", "A", "B"]]

    def __init__(self):
        self.change_path()
        self.config = Common.load_config_file("main_config.toml")
        self.device = self.get_device()
        self.debug = False
        self.set_seed()
        load_dotenv()

    def get_device(self):
        device = "cpu"
        if not self.config.get("use_gpu", True):
            return device

        if torch.backends.mps.is_built():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        return device

    def set_seed(self):
        seed = self.config.get("seed", 0)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def change_path(self):
        root_dir = Path(__file__).parent.parent.parent
        print(f"Common.change_path: chdir to {root_dir}")
        os.chdir(root_dir)

    @staticmethod
    def get_file(file_path):
        src_dir = Path(__file__).parent.parent
        file_path = Path(src_dir, file_path)
        print(f"Common.get_file: {file_path.resolve()}, exists: {file_path.exists()}")
        return file_path

    @staticmethod
    def load_config_file(config_file):
        if not isinstance(config_file, PosixPath):
            config_file = Path(config_file)
        print(
            f"Common.load_config_file: {config_file.resolve()}, exists: {config_file.exists()}"
        )

        if config_file.suffix == ".toml":
            with open(config_file, "rb") as f:
                data = tomli.load(f)
        else:
            data = ConfigParser()
            data.read(str(config_file.resolve()), "UTF-8")
            data = data._sections
        return data


class Logger:
    def __init__(self, common):
        start_tensorboard = common.config.get("start_tensorboard", False)
        start_wandb = common.config.get("start_wandb", False)
        algo = common.config.get("algo")
        env_name = common.config.get("ENV_NAME")
        project_name = f"{env_name}_{algo}"

        # tensorboard
        if start_tensorboard:
            self.writer = SummaryWriter()
        else:
            self.writer = DummyLogger()

        # wandb
        self.wandb_run = None
        if start_wandb:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            self.wandb_run = wandb.init(
                project=project_name,
                sync_tensorboard=start_tensorboard,
                monitor_gym=True,
                config=common.config,
            )

        self.log_dir = self.writer.log_dir
        self.checkpoint_dir = Path(Path.cwd(), "checkpoints/")
        self.actions_dir = Path(self.log_dir, "actions/")
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.actions_dir).mkdir(parents=True)

    def add_scalar(self, name: str, value, step: int = -1):
        self.writer.add_scalar(name, value, step)
        if self.wandb_run:
            wandb.log(data={name: value}, step=step, commit=False)

    def add_histogram(self, name: str, value, step: int = -1):
        self.writer.add_histogram(name, value, step)
        if self.wandb_run:
            wandb.log(data={name: wandb.Histogram(value)}, step=step, commit=False)

    def add_hparams(self, hparams: dict, metrics: dict = None):
        if metrics is None:
            metrics = {}
        self.writer.add_hparams(hparams, metrics)

    def add_file(self, file_path):
        from_path = Path(file_path)
        to_path = Path(self.log_dir, from_path.name)
        print(f"Logger.add_file: from: {from_path.resolve()}, {from_path.exists()})")
        print(f"Logger.add_file: to: {to_path.resolve()}")

        shutil.copy(from_path, to_path)
        if self.wandb_run:
            artifact = wandb.Artifact(name=from_path.name, type=from_path.suffix)
            artifact.add_file(local_path=str(from_path), name=from_path.name)
            self.wandb_run.log_artifact(artifact)

    def add_figure(self, name, fig: plt.Figure):
        self.writer.add_figure(name, fig)
        if self.wandb_run:
            wandb.log(data={name: wandb.Image(fig)}, commit=False)

    def add_pickle(self, name, element):
        to_path = Path(self.log_dir, f"{name}.pkl")
        with open(to_path, "wb") as f:
            pickle.dump(element, f, protocol=5)
        if self.wandb_run:
            artifact = wandb.Artifact(name=name, type="pkl")
            artifact.add_file(local_path=str(to_path), name=name)
            self.wandb_run.log_artifact(artifact)

    def flush(self):
        self.writer.flush()
        if self.wandb_run:
            wandb.log(data={}, commit=True)

    def close(self):
        self.flush()
        self.writer.close()
        if self.wandb_run:
            wandb.finish()


class DummyLogger:
    def __init__(self):
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(Path.cwd(), "runs", date_str)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = Path(self.log_dir, "run.log")
        self.log_file = open(log_file_path, "w+")

    def add_scalar(self, name: str, value, step: int):
        self.log_file.write(f"step: {step}, {name}: {value}\n")

    def add_histogram(self, name: str, value, step: int):
        pass

    def add_hparams(self, hparams: dict, metrics: dict):
        pass

    def add_figure(self, name, fig: plt.Figure):
        fig.savefig(Path(self.log_dir, f"{name}.svg"))

    def flush(self):
        if self.log_file:
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()


class Tracker:
    def __init__(self, algo: str, logger: Logger):
        self.algo = algo
        self.logger = logger
        # best accumulated reward so far
        self.best_reward = -math.inf

    # def store_action(self, info: dict, episode: int):
    #     reward = info.get("reward")
    #     normalized_reward = info.get("normalized_reward")
    #     self.rewards += reward
    #     self.logger.add_scalar("reward", reward, episode)
    #     self.logger.add_scalar("normalized_reward", normalized_reward, episode)

    def end_of_episode(self, agent, info: dict, episode: int):
        if not info or not info.get("episode"):
            return

        get_flag = info.get("flag_get", False)
        episode_info = info.get("episode")
        reward = episode_info["r"]
        len = episode_info["l"]
        avg = reward / len
        time = episode_info["t"]
        actions = episode_info["a"]

        # store best reward so far
        if reward > self.best_reward:
            self.best_reward = reward
            self.save_actions(actions, reward, episode)

        # log some vars
        print(
            f"# ep {episode}, r: {reward:.0f} (best:{self.best_reward:.0f}), avg: {avg:.2f}, len: {len}, time: {time:.0f}"
        )
        self.logger.add_scalar("get_flag", get_flag, episode)
        self.logger.add_scalar("episodic_return", reward, episode)
        self.logger.add_scalar("episodic_length", len, episode)

        # log agent LR
        if agent.scheduler and agent.scheduler.get_last_lr():
            current_lr = agent.scheduler.get_last_lr()[0]
            self.logger.add_scalar("learning_rate", current_lr, episode)

    def save_actions(self, actions, rewards, episode):
        path = Path(
            self.logger.actions_dir,
            f"{self.algo}_actions_ep:{episode}_rw:{int(rewards)}.pt",
        )
        if not self.logger.actions_dir.exists():
            print(f"WARNING save_actions: path doesn't exists, skipping save: {path}")
        else:
            torch.save(np.array(actions), path)
