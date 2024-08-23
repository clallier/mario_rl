import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam


class DPOAgent:

    def __init__(self, common, sim, dpo_config):
        self.common = common
        self.sim = sim
        nn_input_size = np.array(sim.env.single_observation_space.shape).prod()
        nn_output_size = sim.env.single_action_space.n
        self.network = nn.Sequential(
            # input: 4 4, 30, 30
            # in_channel, out_channel, kernel_size, stride, padding
            nn.Conv2d(4, 32, 3, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            # nn.Conv2d(64, 64, 3, stride=1),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, nn_output_size)
        self.critic = nn.Linear(128, 1)
        #
        # self.critic = nn.Sequential(
        #     nn.Linear(nn_input_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        # self.actor = nn.Sequential(
        #     nn.Linear(nn_input_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, nn_output_size)
        # )

        self.network.to(self.common.device)
        self.critic.to(self.common.device)
        self.actor.to(self.common.device)

        lr = dpo_config.get("lr", 0.001)
        self.optim = Adam(self.actor.parameters(), lr=lr, eps=1e-5)

    def get_critic_value(self, x):
        return self.critic(self.network(x / 255.0))
        # return self.critic(x)

    def get_action_and_critic(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        # logits = self.actor(x)
        probs = Categorical(logits)
        probs.probs = torch.clamp(probs.probs, 0)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
        # return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def set_lr(self, lrnow):
        self.optim.param_groups[0]['lr'] = lrnow

    def retropropagate(self, loss, max_grad_norm):
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.optim.step()
