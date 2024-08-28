# Adapted from Sourish Kundu's video : https://www.youtube.com/watch?v=_gmQZToTMac

import numpy as np
from tensordict import TensorDict
import torch
from src.Common.common import Tracker
from src.Common.conv_calc import debug_count_params, debug_nn_size
from src.Common.async_single_sim import AsyncSingleSim
from src.Common.trainer import Trainer
from src.DQNN.dqnn_agent import DQNNAgent
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from pathlib import Path


class DQNNTrainer(Trainer):
    def init(self):
        self.tracker = Tracker(self.logger)

        # Replay buffer
        self.replay_buffer_capacity = self.config.get("replay_buffer_capacity", 100_000)
        self.storage_dir = Path(Path.cwd(), "_dump")
        print(
            "### storage_dir: ", self.storage_dir.resolve(), self.storage_dir.exists()
        )
        storage = LazyMemmapStorage(
            self.replay_buffer_capacity, scratch_dir=self.storage_dir
        )
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

        if self.debug:
            state = self.sim.reset()
            debug_nn_size(self.agent.online_network.network, state, self.device)
            debug_count_params(self.agent.online_network.network)

    def create_sim(self):
        return AsyncSingleSim(self.common)

    def create_agent(self):
        nn_hidden_size = self.config.get("nn_hidden_size", 512)
        nn_output_size = self.sim.single_action_space.n
        return DQNNAgent(
            nn_hidden_size, nn_output_size, self.config, self.common, self.logger
        )

    def run_episode(self, episode):
        done = False
        state = self.sim.reset()
        self.tracker.init_reward()

        while not done:
            action = self.agent.get_action(state)
            # print("action: ", action)
            next_state, reward, done, info = self.sim.step(action)
            self.tracker.store_action(action, info, self.agent.learn_step_counter)
            self.store_in_memory(state, action, info, next_state, done)
            self.agent.learn(self.replay_buffer)
            state = next_state

            if self.debug:
                self.sim.render()

        # end of current episode
        current_lr = self.agent.scheduler.get_last_lr()[0]
        self.logger.add_scalar(
            "learning_rate", current_lr, self.agent.learn_step_counter
        )

        self.tracker.end_of_episode(info, episode, self.save_actions)

    def save_complete_state(self, path: Path):
        torch.save(
            {
                "episode": self.episode,
                "epsilon": self.agent.epsilon,
                "optimizer": self.agent.optimizer.state_dict(),
                "scheduler": self.agent.scheduler.state_dict(),
                "online_network": self.agent.online_network.state_dict(),
                "sim": self.sim.state_dict(),
            },
            path,
        )
        replay_buffer_path = Path(path.parent, path.stem, "replay_buffer")
        replay_buffer_path.mkdir(parents=True, exist_ok=True)
        self.replay_buffer.dumps(replay_buffer_path)
        print("### replay buffer size: ", len(self.replay_buffer))

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

    def load_complete_state(self, path):
        replay_buffer_path = Path(path.parent, path.stem, "replay_buffer")
        self.replay_buffer.loads(replay_buffer_path)
        print("### replay buffer size: ", len(self.replay_buffer))

        load_state = torch.load(path, weights_only=False)
        self.episode = load_state["episode"]
        self.agent.epsilon = load_state["epsilon"]
        self.agent.optimizer.load_state_dict(load_state["optimizer"])
        self.agent.scheduler.load_state_dict(load_state["scheduler"])
        model_state_dict = load_state["online_network"]
        self.agent.online_network.load_state_dict(model_state_dict)
        # sync networks
        self.agent.target_network.load_state_dict(model_state_dict)
        self.sim.load_state_dict(load_state["sim"])

    def save_actions(self, actions, episode, rewards):
        path = Path(
            self.logger.actions_dir, f"agent_actions_ep:{episode}_rw:{int(rewards)}.pt"
        )
        if not self.logger.actions_dir.exists():
            print(f"WARNING save_actions: path doesn't exists, skipping save: {path}")
        else:
            torch.save(np.array(actions), path)
