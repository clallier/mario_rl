from torch.distributions import Categorical
from torch import nn
from torch.optim import AdamW, lr_scheduler


class ReinforceAgent:

    def __init__(
        self,
        common,
        nn_input_size,
        nn_output_size,
        config: dict,
    ):
        self.common = common
        self.device = common.device
        self.config = config

        # num episodes
        self.num_episodes = self.config.get("NUM_OF_EPISODES")
        # learning_rate
        self.learning_rate = self.config.get("learning_rate")
        self.lr_start_factor = self.config.get("learning_rate_start_factor")
        self.lr_end_factor = self.config.get("learning_rate_end_factor")

        hidden_size = nn_input_size // 2
        hidden_size_2 = hidden_size // 2
        hidden_size_4 = hidden_size_2 // 2
        self.nn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nn_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_4),
            nn.ReLU(),
            nn.Linear(hidden_size_4, nn_output_size),
            nn.Softmax(dim=-1),
        )
        self.nn.to(self.device)

        self.optimizer = AdamW(
            self.nn.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        self.scheduler = lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.lr_start_factor,
            end_factor=self.lr_end_factor,
            total_iters=self.num_episodes,
        )

    def get_action(self, x, action=None):
        probs = self.nn(x)
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def retropropagate(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
