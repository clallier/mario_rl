import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import AdamW, lr_scheduler


class DPOAgent:

    def __init__(self, common, nn_hidden_size, nn_output_size, config):
        self.common = common
        self.config = config
        self.num_episodes = self.config.get("NUM_OF_EPISODES")
        self.learning_rate = self.config.get("learning_rate")
        self.lr_start_factor = self.config.get("learning_rate_start_factor")
        self.lr_end_factor = self.config.get("learning_rate_end_factor")

        self.network = nn.Sequential(
            # input: envs, channels, time, w,  h
            # input: 4,    1,        4,    30, 30
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3, 3),
                stride=(3, 2, 2),
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, nn_hidden_size),
            nn.ReLU(),
        )

        hidden_size_2 = nn_hidden_size // 2
        hidden_size_4 = hidden_size_2 // 2
        self.critic = nn.Sequential(
            self.network,
            nn.Linear(nn_hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_4),
            nn.ReLU(),
            nn.Linear(hidden_size_4, 1),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            self.network,
            nn.Linear(nn_hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_4),
            nn.ReLU(),
            nn.Linear(hidden_size_4, nn_output_size),
            nn.ReLU(),
        )

        self.critic.to(self.common.device)
        self.actor.to(self.common.device)

        self.optimizer = AdamW(
            self.actor.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )

        self.scheduler = lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.lr_start_factor,
            end_factor=self.lr_end_factor,
            total_iters=self.num_episodes,
        )

    def get_critic_value(self, x):
        return self.critic(x)

    def get_action_and_critic(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits)
        dist.probs = torch.clamp(dist.probs, 0)
        if action is None:
            action = dist.sample()
        critic_values = self.critic(x)
        return action, dist.log_prob(action), dist.entropy(), critic_values

    def retropropagate(self, loss, max_grad_norm):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
