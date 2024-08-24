import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam


class DPOAgent:

    def __init__(self, common, nn_hidden_size, nn_output_size, dpo_config):
        self.common = common
        self.network = nn.Sequential(
            # input: envs, channels, time, w,  h
            # input: 4,    1,        4,    30, 30
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), stride=(3, 2, 2), padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, nn_hidden_size),
            nn.ReLU(),
        )
        
        hidden_size_2 = nn_hidden_size // 2
        hidden_size_4 = hidden_size_2 // 2
        self.critic = nn.Sequential(
            nn.Linear(nn_hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_4),
            nn.ReLU(),
            nn.Linear(hidden_size_4, 1),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(nn_hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_4),
            nn.ReLU(),
            nn.Linear(hidden_size_4, nn_output_size),
            nn.ReLU()
        )

        self.network.to(self.common.device)
        self.critic.to(self.common.device)
        self.actor.to(self.common.device)

        lr = dpo_config.get("lr", 0.001)
        self.optim = Adam(self.actor.parameters(), lr=lr, eps=1e-5)

    def get_critic_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_critic(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits)
        probs.probs = torch.clamp(probs.probs, 0)
        if action is None:
            action = probs.sample()
        critic_values = self.critic(hidden)
        return action, probs.log_prob(action), probs.entropy(), critic_values 

    def set_lr(self, lrnow):
        self.optim.param_groups[0]['lr'] = lrnow

    def retropropagate(self, loss, max_grad_norm):
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.optim.step()
