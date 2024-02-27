from pathlib import Path

import numpy as np
import tomli
import torch
from tensordict import TensorDict
from torch import nn
from torch.optim import lr_scheduler
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from common import Common, Logger
from src.DQNN.agent_nn import AgentNN


class DQNNAgent:
    def __init__(self, input_dims, output_dims, common: Common, logger: Logger):
        self.common = common
        self.logger = logger
        self.config = self.load_config_file('config.toml')
        self.num_actions = output_dims
        self.learn_step_counter = 0
        # discount factor
        self.gamma = self.config['gamma']
        # batch size
        self.batch_size = self.config['batch_size']
        # exploration rate
        self.epsilon = self.config['epsilon_init']
        # Networks
        self.online_network = AgentNN(
            input_dims, output_dims, device=self.common.device)
        self.target_network = AgentNN(
            input_dims, output_dims, freeze=True, device=self.common.device)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config['learning_rate'])

        self.scheduler = lr_scheduler.LinearLR(self.optimizer,
                                               start_factor=self.config['learning_rate_start_factor'],
                                               end_factor=self.config['learning_rate_end_factor'],
                                               total_iters=self.common.NUM_OF_EPISODES)
        # self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.learning_rate_decay)

        self.loss = nn.SmoothL1Loss()  # Huber loss # nn.MSELoss()

        # Replay buffer
        self.replay_buffer_capacity = 100_000
        self.storage_dir = Path(Path.cwd(), "_dump")
        print("### storage_dir: ", self.storage_dir.resolve(), self.storage_dir.exists())
        storage = LazyMemmapStorage(
            self.replay_buffer_capacity, scratch_dir=self.storage_dir)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def load_config_file(self, load_config_file):
        config_file = Path(Path(__file__).parent, load_config_file)
        print(config_file.resolve(), config_file.exists())
        config_file = tomli.loads(config_file.read_text(encoding="utf-8"))
        print(f"DDQNAgent.load_config_file: {config_file}")
        return config_file

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32)
            state = state.unsqueeze(0).to(self.common.device)
            q_values = self.online_network(state)
            return q_values.argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.config['epsilon_decay'], self.config['epsilon_min'])

    def save_state(self, path):
        torch.save({
            'epsilon': self.epsilon,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'online_network': self.online_network.state_dict(),
        },
            path
        )
        print("### replay buffer size: ", len(self.replay_buffer))

    def load_state(self, path):
        storage = LazyMemmapStorage(
            self.replay_buffer_capacity, scratch_dir=self.storage_dir)
        print("### loading buffer storage", len(storage))
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

        load_state = torch.load(path)
        self.epsilon = load_state['epsilon']
        self.optimizer.load_state_dict(load_state['optimizer'])
        self.scheduler.load_state_dict(load_state['scheduler'])
        model_state_dict = load_state['online_network']
        self.online_network.load_state_dict(model_state_dict)
        # force sync networks?
        self.target_network.load_state_dict(model_state_dict)

    def store_in_memory(self, state, action, reward, next_state, done):
        if isinstance(reward, dict) and 'normalized' in reward:
            reward = reward['normalized'] 

        self.replay_buffer.add(TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.float32),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32),
            "done": torch.tensor(done)
        }, batch_size=[]))

    def sync_networks(self):
        if self.learn_step_counter % self.config['sync_network_rate'] == 0 and self.learn_step_counter > 0:
            self.logger.add_scalar("sync", 1, self.learn_step_counter)
            self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self, episode):
        # if not enough samples in replay buffer, do nothing
        if len(self.replay_buffer) < self.batch_size:
            return

        # if needed, sync target network with online network
        self.sync_networks()

        # reset gradients
        self.optimizer.zero_grad()
        # get some samples from replay buffer
        samples = self.replay_buffer.sample(self.batch_size).to(self.common.device)

        # get q values for current state
        keys = ('state', 'action', 'reward', 'next_state', 'done')

        states, actions, rewards, next_states, dones = [
            samples[key] for key in keys]

        # Shape is (batch_size, n_actions)
        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[
            np.arange(self.batch_size), actions.squeeze()
        ]

        # target q values
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = \
            rewards + self.gamma * target_q_values * (1 - dones.float())

        # compute loss based on the difference between predicted and
        # target q values because we want to minimize this difference:
        # predicted_q_values is the output of the network,
        # target_q_values is the target (the value we want the network
        # to output, computed from the Bellman equation)
        loss = self.loss(predicted_q_values, target_q_values)
        self.logger.add_scalar("loss", loss, self.learn_step_counter)
        loss.backward()

        self.optimizer.step()

        self.decay_epsilon()
        self.logger.add_scalar("epsilon", self.epsilon, self.learn_step_counter)
        self.learn_step_counter += 1

    def debug_nn_size(self, state, device='mps'):
        x0 = torch.as_tensor(np.array(state)).float().to(device)
        print("input shape:", x0.shape)

        with torch.no_grad():
            x = x0
            for layer in self.online_network.conv:
                x = layer(x)
                print(type(layer), x.shape)
            params = (sum(
                p.numel() for p in self.online_network.network.parameters() if p.requires_grad))
            print("params: ", params)
