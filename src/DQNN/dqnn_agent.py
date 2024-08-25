from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torch.optim import lr_scheduler
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from src.Common.common import Common, Logger
from src.DQNN.agent_nn import AgentNN


class DQNNAgent:
    def __init__(
        self,
        nn_hidden_size: int,
        nn_output_size: int,
        ddqn_config: dict,
        common: Common,
        logger: Logger,
    ):
        self.logger = logger
        self.config = ddqn_config
        self.num_actions = nn_output_size
        self.learn_step_counter = 0
        # device
        self.device = common.device
        # num episodes
        self.num_episodes = self.config.get("NUM_OF_EPISODES")
        # discount factor
        self.gamma = self.config.get("gamma")
        # batch size
        self.batch_size = self.config.get("batch_size")
        # exploration rate
        self.epsilon = self.config.get("epsilon_init")
        # learning_rate
        self.learning_rate = self.config.get("learning_rate")
        self.lr_start_factor = self.config.get("learning_rate_start_factor")
        self.lr_end_factor = self.config.get("learning_rate_end_factor")
        # epsilon
        self.epsilon_decay = self.config.get("epsilon_decay")
        self.epsilon_min = self.config.get("epsilon_min")
        # sync_network_rate
        self.sync_network_rate = self.config.get("sync_network_rate")

        # keys
        self.keys = ("state", "action", "reward", "next_state", "done")

        # Networks
        self.online_network = AgentNN(
            nn_hidden_size, nn_output_size, device=self.device
        )
        self.target_network = AgentNN(
            nn_hidden_size, nn_output_size, freeze=True, device=self.device
        )

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.learning_rate
        )

        self.scheduler = lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.lr_start_factor,
            end_factor=self.lr_end_factor,
            total_iters=self.num_episodes,
        )
        # self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.learning_rate_decay)

        self.loss = nn.SmoothL1Loss()  # Huber loss # nn.MSELoss()

        # Replay buffer
        self.replay_buffer_capacity = 100_000
        self.storage_dir = Path(Path.cwd(), "_dump")
        print(
            "### storage_dir: ", self.storage_dir.resolve(), self.storage_dir.exists()
        )
        storage = LazyMemmapStorage(
            self.replay_buffer_capacity, scratch_dir=self.storage_dir
        )
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
            q_values = self.online_network(state)
            return q_values.argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_state(self, path):
        torch.save(
            {
                "epsilon": self.epsilon,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "online_network": self.online_network.state_dict(),
            },
            path,
        )
        print("### replay buffer size: ", len(self.replay_buffer))

    def load_state(self, path):
        storage = LazyMemmapStorage(
            self.replay_buffer_capacity, scratch_dir=self.storage_dir
        )
        print("### loading buffer storage", len(storage))
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

        load_state = torch.load(path)
        self.epsilon = load_state["epsilon"]
        self.optimizer.load_state_dict(load_state["optimizer"])
        self.scheduler.load_state_dict(load_state["scheduler"])
        model_state_dict = load_state["online_network"]
        self.online_network.load_state_dict(model_state_dict)
        # force sync networks?
        self.target_network.load_state_dict(model_state_dict)

    def store_in_memory(self, state, action, info, next_state, done):
        reward = info.get("reward")
        state = np.squeeze(state, axis=0)
        next_state = np.squeeze(next_state, axis=0)

        self.replay_buffer.add(
            TensorDict(
                {
                    "state": torch.tensor(state, dtype=torch.float32),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "next_state": torch.tensor(next_state, dtype=torch.float32),
                    "done": torch.tensor(done),
                },
                batch_size=[],
            )
        )

    def sync_networks(self):
        if (
            self.learn_step_counter % self.sync_network_rate == 0
            and self.learn_step_counter > 0
        ):
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
        samples = self.replay_buffer.sample(self.batch_size).to(self.device)

        # get q values for current state
        states, actions, rewards, next_states, dones = [
            samples[key] for key in self.keys
        ]

        # Shape is (batch_size, n_actions)
        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[
            np.arange(self.batch_size), actions.squeeze()
        ]

        # target q values
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

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
