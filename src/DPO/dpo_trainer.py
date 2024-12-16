import time

import numpy as np
import torch

from src.Common.async_multi_sim import AsyncMultiSims
from src.Common.trainer import Trainer
from src.DPO.dpo_agent import DPOAgent
from src.Common.conv_calc import debug_count_params


# from https://www.youtube.com/watch?v=MEt6rrxH8W4
class DPOTrainer(Trainer):
    def init(self):
        self.num_episodes = self.config.get("NUM_OF_EPISODES", 10_000)
        self.num_steps_per_update = self.config.get("num_steps_per_update", 64)
        self.batch_size = self.num_envs * self.num_steps_per_update

        self.obs_space_shape = (1,) + self.sim.single_observation_space.shape
        self.action_space_shape = self.sim.single_action_space.shape
        self.update_epochs = self.config.get("update_epochs")
        num_minibatches = self.config.get("num_minibatches")
        self.minibatch_size = int(self.batch_size // num_minibatches)
        self.clip_coef = self.config.get("clip_coef")
        self.norm_adv = self.config.get("norm_adv")
        self.clip_vloss = self.config.get("clip_vloss")
        self.ent_coef = self.config.get("ent_coef")
        self.vf_coef = self.config.get("vf_coef")
        self.max_grad_norm = self.config.get("max_grad_norm")
        self.target_kl = self.config.get("target_kl")
        self.gamma = self.config.get("gamma")

        device = self.device
        steps_envs_shape = (self.num_steps_per_update, self.num_envs)
        self.obs = torch.zeros(steps_envs_shape + self.obs_space_shape).to(device)
        self.info = torch.zeros(steps_envs_shape + self.info_shape).to(device)
        self.actions = torch.zeros(steps_envs_shape + self.action_space_shape).to(
            device
        )
        self.log_probs = torch.zeros(steps_envs_shape).to(device)
        self.rewards = torch.zeros(steps_envs_shape).to(device)
        self.dones = torch.zeros(steps_envs_shape).to(device)
        self.critic_values = torch.zeros(steps_envs_shape).to(device)

        self.update = 0
        self.global_step = 0
        self.start_time = time.time()
        self.next_obs = self.to_tensor(self.sim.reset())
        self.next_info = torch.zeros((self.num_envs,) + self.info_shape).to(device)
        self.next_done = torch.zeros(self.num_envs).to(device)
        self.num_updates = self.num_episodes // self.batch_size

        if self.debug:
            # state = self.sim.reset()
            # debug_nn_size(self.agent.actor, state, device)
            debug_count_params(self.agent.network)
            debug_count_params(self.agent.actor)

    def create_sim(self):
        self.num_envs = self.config.get("num_envs", 1)
        return AsyncMultiSims(self.common, self.num_envs)

    def create_agent(self):
        nn_hidden_size = self.config.get("nn_hidden_size", 128)
        nn_output_size = self.sim.single_action_space.n
        nn_info_size = self.info_shape[0]
        return DPOAgent(
            self.common, nn_info_size, nn_hidden_size, nn_output_size, self.config
        )

    def load_complete_state(self, path):
        load_state = torch.load(path, weights_only=False)
        self.update = load_state["update"]
        self.global_step = load_state["global_step"]
        self.obs = load_state["obs"]
        self.info = load_state["info"]
        self.log_probs = load_state["log_probs"]
        self.critic_values = load_state["critic_values"]
        self.actions = load_state["actions"]
        self.agent.optimizer.load_state_dict(load_state["optimizer"])
        self.agent.scheduler.load_state_dict(load_state["scheduler"])
        self.agent.network.load_state_dict(load_state["network"])
        self.agent.critic.load_state_dict(load_state["critic"])
        self.agent.actor.load_state_dict(load_state["actor"])
        self.sim.load_state_dict(load_state["sim"])

    def save_complete_state(self, path):
        torch.save(
            {
                "update": self.update,
                # episode
                "global_step": self.global_step,
                "obs": self.obs,
                "info": self.info,
                "log_probs": self.log_probs,
                "critic_values": self.critic_values,
                "actions": self.actions,
                "optimizer": self.agent.optimizer.state_dict(),
                "scheduler": self.agent.scheduler.state_dict(),
                "network": self.network.critic.state_dict(),
                "critic": self.agent.critic.state_dict(),
                "actor": self.agent.actor.state_dict(),
                "sim": self.sim.state_dict(),
            },
            path,
        )

    def train_init(self):
        return super().train_init()

    def run_episode(self) -> dict:
        return super().run_episode()

    def train(self):
        from_update = self.update + 1
        for update in range(from_update, self.num_updates + 1):
            self.update = update
            for step in range(self.num_steps_per_update):
                self.global_step += self.num_envs
                self.obs[step] = self.next_obs
                self.info[step] = self.next_info
                self.dones[step] = self.next_done

                # rollouts (no need for gradient)
                with torch.no_grad():
                    # 7, 1, 4, 30, 30; 7
                    actions, log_probs, entropies, critic_values = (
                        self.agent.get_action_and_critic(self.next_obs, self.next_info)
                    )
                self.actions[step] = actions
                self.log_probs[step] = log_probs
                self.critic_values[step] = critic_values.flatten()

                # step the envs
                next_obs, reward, done, info = self.sim.step(actions.cpu().numpy())
                rewards = np.array([i.get("reward") for i in info])
                print(f"r: {reward.item():.3f}")
                self.rewards[step] = self.to_tensor(rewards)
                self.next_obs = self.to_tensor(next_obs)
                self.next_info = self.info_to_tensor(info)
                self.next_done = self.to_tensor(done)

                if self.debug:
                    self.sim.render()

                for item in info:
                    if "episode" in item.keys():
                        self.log_end_of_episode(item["episode"])
            self.optimize_policy()
            self.logger.flush()
            # self.save_state()
        self.close()

    def log_end_of_episode(self, episode_info: dict):
        reward = episode_info["r"]
        len = episode_info["l"]
        avg = reward / len
        time = episode_info["t"]
        print(
            f"# ep {self.global_step}, r: {reward:.2f}, avg: {avg:.2f}, len: {len}, time: {time:.0f}"
        )
        self.logger.add_scalar("episodic_return", reward, self.global_step)
        self.logger.add_scalar("episodic_length", len, self.global_step)

    def optimize_policy(self):
        self.sim.reset()
        # compute returns and avantages
        with torch.no_grad():
            next_values = self.agent.get_critic_value(
                self.next_obs, self.next_info
            ).reshape(1, -1)
            returns = torch.zeros_like(self.rewards).to(self.device)
            for t in reversed(range(self.num_steps_per_update)):
                if t == self.num_steps_per_update - 1:
                    next_non_terminal = 1.0 - self.next_done
                    next_return = next_values
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = (
                    self.rewards[t] + self.gamma * next_non_terminal * next_return
                )
            advantages = returns - self.critic_values

        v_loss = 0
        pg_loss = 0
        entropy_loss = 0

        # flatten the batch
        flatten_obs = self.obs.reshape((-1,) + self.obs_space_shape)
        flatten_info = self.info.reshape((-1,) + self.info_shape)
        flatten_log_probs = self.log_probs.reshape(-1)
        flatten_actions = self.actions.reshape((-1,) + self.action_space_shape)
        flatten_advantages = advantages.reshape(-1)
        flatten_returns = returns.reshape(-1)
        flatten_values = self.critic_values.reshape(-1)

        # Optimizing the policy and value network
        rnd_idx = np.arange(self.batch_size)
        clip_fracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(rnd_idx)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                sub_idx = rnd_idx[start:end]

                obs = flatten_obs[sub_idx]
                info = flatten_info[sub_idx]
                actions = flatten_actions[sub_idx]
                new_action, new_log_prob, entropy, new_values = (
                    self.agent.get_action_and_critic(obs, info, actions)
                )
                log_ratio = new_log_prob - flatten_log_probs[sub_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                sub_advantages = flatten_advantages[sub_idx]
                if self.norm_adv:
                    sub_advantages = (sub_advantages - sub_advantages.mean()) / (
                        sub_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -sub_advantages * ratio
                pg_loss2 = -sub_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                # TODO check this algorithm
                pg_loss = pg_loss.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (pg_loss - flatten_returns[sub_idx]) ** 2
                    v_clipped = flatten_values[sub_idx] + torch.clamp(
                        pg_loss - flatten_values[sub_idx],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - flatten_returns[sub_idx]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((pg_loss - flatten_returns[sub_idx]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                loss = pg_loss1.sum()
                print("loss", loss.item())
                self.agent.retropropagate(loss, self.max_grad_norm)

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

            y_pred, y_true = flatten_values.cpu().numpy(), flatten_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )
            print("explained_var", explained_var.item())

            self.logger.add_scalar(
                "charts/learning_rate",
                self.agent.optimizer.param_groups[0]["lr"],
                self.global_step,
            )
            self.logger.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.logger.add_scalar(
                "losses/policy_loss", pg_loss.item(), self.global_step
            )
            self.logger.add_scalar(
                "losses/entropy_loss", entropy_loss.item(), self.global_step
            )
            self.logger.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), self.global_step
            )
            self.logger.add_scalar(
                "losses/approx_kl", approx_kl.item(), self.global_step
            )
            self.logger.add_scalar(
                "losses/clip_frac", np.mean(clip_fracs), self.global_step
            )
            self.logger.add_scalar(
                "losses/explained_variance", explained_var, self.global_step
            )
            self.logger.add_scalar(
                "charts/SPS",
                int(self.global_step / (time.time() - self.start_time)),
                self.global_step,
            )

    def eval(self):
        num_eval = 5
        eval_step = self.global_step

        for update in range(num_eval):
            eval_step += 1
            all_done = False
            all_info = dict((i, {}) for i in range(self.num_envs))

            while all_done is False:
                # rollouts (no need for gradient)
                with torch.no_grad():
                    action, logprob, entropy, crit_value = (
                        self.agent.get_action_and_critic(self.next_obs)
                    )

                # step the envs
                next_obs, reward, done, info = self.sim.step(action.cpu().numpy())

                for item, i in zip(info, all_info):
                    if "episode" not in all_info[i].keys() and "episode" in item.keys():
                        all_info[i] = item
                all_done = all([len(all_info[i].keys()) > 0 for i in all_info])

            avg_episodic_return = (
                sum([all_info[i]["episode"]["r"] for i in all_info]) / self.num_envs
            )
            avg_episodic_length = (
                sum([all_info[i]["episode"]["l"] for i in all_info]) / self.num_envs
            )
            self.logger.add_scalar(
                "eval/avg_episodic_return", avg_episodic_return, eval_step
            )
            self.logger.add_scalar(
                "eval/avg_episodic_length", avg_episodic_length, eval_step
            )
            self.logger.flush()
