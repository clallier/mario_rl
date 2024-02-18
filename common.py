from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


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
            self.writer.add_scalar(name , value, step)
