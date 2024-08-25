import time

import numpy as np
import torch

from src.Common.common import Common, Logger
from src.Common.async_multi_sim import AsyncMultiSims
from src.DPO.dpo_agent import DPOAgent


# from https://www.youtube.com/watch?v=MEt6rrxH8W4
class DPOTrainer:
    def __init__(self, common):
        self.common = common
        self.logger = Logger(common)

        # DPO config
        dpo_config_path = Common.get_file(f"./config_files/{common.config.get('dpo_config_file')}")
        self.dpo_config = common.load_config_file(dpo_config_path)
        self.logger.add_file(dpo_config_path)

        self.total_timesteps = self.dpo_config.get('total_timesteps', 10_000)
        self.num_steps = self.dpo_config.get('num_steps', 64)
        self.num_envs = self.dpo_config.get('num_envs', 1)
        self.batch_size = self.num_envs * self.num_steps
        self.anneal_lr = self.dpo_config.get('anneal_lr', True)
        self.lr = self.dpo_config.get('lr', 0.001)

        self.sims = AsyncMultiSims(common, self.num_envs)

        obs_space_shape = self.sims.single_observation_space.shape
        action_space_shape = self.sims.single_action_space.shape
        action_space_n = self.sims.single_action_space.n
        device = self.common.device

        steps_envs_shape = (self.num_steps, self.num_envs)
        self.obs = torch.zeros(steps_envs_shape + obs_space_shape).to(device)
        self.actions = torch.zeros(steps_envs_shape + action_space_shape).to(device)
        self.log_probs = torch.zeros(steps_envs_shape).to(device)
        self.rewards = torch.zeros(steps_envs_shape).to(device)
        self.dones = torch.zeros(steps_envs_shape).to(device)
        self.values = torch.zeros(steps_envs_shape).to(device)

        self.global_step = 0
        self.start_time = time.time()
        self.next_obs = torch.tensor(self.sims.reset()).to(device)
        self.next_done = torch.zeros(self.num_envs).to(device)
        self.num_updates = self.total_timesteps // self.batch_size

        nn_hidden_size = self.dpo_config.get('nn_hidden_size', 128)
        nn_output_size = action_space_n
        self.agent = DPOAgent(common, nn_hidden_size, nn_output_size, self.dpo_config)

        self.train()
        self.eval()
        self.close()

    def train(self):
        device = self.common.device

        for update in range(1, self.num_updates + 1):
            if self.anneal_lr and self.num_updates > 0:
                frac = 1.0 - (update - 1) / self.num_updates
                lr_now = frac * self.lr
                self.agent.set_lr(lr_now)

            for step in range(self.num_steps):
                self.global_step += self.num_envs
                self.obs[step] = self.next_obs
                self.dones[step] = self.next_done

                # rollouts (no need for gradient)
                with torch.no_grad():
                    # 7, 1, 4, 30, 30
                    obs = self.next_obs.unsqueeze(1)
                    actions, log_probs, entropies, critic_values = self.agent.get_action_and_critic(obs)
                self.actions[step] = actions
                self.log_probs[step] = log_probs
                self.values[step] = critic_values.flatten()

                # step the envs
                next_obs, reward, done, info = self.sims.step(actions.cpu().numpy())
                self.rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)
                self.next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
                self.next_done = torch.tensor(done, dtype=torch.float32).to(device)

                for item in info:
                    if "episode" in item.keys():
                        e = item["episode"]
                        self.logger.add_scalar("episodic_return", e["r"], self.global_step)
                        self.logger.add_scalar("episodic_length", e["l"], self.global_step)

            self.optimize_policy()
            self.logger.flush()

    def optimize_policy(self):
        device = self.common.device
        num_steps = self.num_steps
        gamma = self.dpo_config.get("gamma")

        # bootstrap value if not done
        with torch.no_grad():
            obs = self.next_obs.unsqueeze(1)
            next_value = self.agent.get_critic_value(obs).reshape(1, -1)
            returns = torch.zeros_like(self.rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - self.next_done
                    next_return = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = self.rewards[t] + gamma * next_non_terminal * next_return
            advantages = returns - self.values

        obs_space_shape = self.sims.single_observation_space.shape
        action_space_shape = self.sims.single_action_space.shape
        update_epochs = self.dpo_config.get("update_epochs")
        num_minibatches = self.dpo_config.get("num_minibatches")
        minibatch_size = int(self.batch_size // num_minibatches)
        clip_coef = self.dpo_config.get("clip_coef")
        norm_adv = self.dpo_config.get("norm_adv")
        clip_vloss = self.dpo_config.get("clip_vloss")
        ent_coef = self.dpo_config.get("ent_coef")
        vf_coef = self.dpo_config.get("vf_coef")
        max_grad_norm = self.dpo_config.get("max_grad_norm")
        target_kl = self.dpo_config.get("target_kl")

        v_loss = 0
        pg_loss = 0
        entropy_loss = 0

        # flatten the batch
        flatten_obs = self.obs.reshape((-1,) + obs_space_shape)
        flatten_log_probs = self.log_probs.reshape(-1)
        flatten_actions = self.actions.reshape((-1,) + action_space_shape)
        flatten_advantages = advantages.reshape(-1)
        flatten_returns = returns.reshape(-1)
        flatten_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        rnd_idx = np.arange(self.batch_size)
        clip_fracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(rnd_idx)
            for start in range(0, self.batch_size, minibatch_size):
                end = start + minibatch_size
                sub_idx = rnd_idx[start:end]

                obs = flatten_obs[sub_idx].unsqueeze(1)
                actions = flatten_actions.long()[sub_idx]
                new_action, new_log_prob, entropy, new_value = self.agent.get_action_and_critic(obs, actions)
                log_ratio = new_log_prob - flatten_log_probs[sub_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                sub_advantages = flatten_advantages[sub_idx]
                if norm_adv:
                    sub_advantages = (sub_advantages - sub_advantages.mean()) / (sub_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -sub_advantages * ratio
                pg_loss2 = -sub_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (new_value - flatten_returns[sub_idx]) ** 2
                    v_clipped = flatten_values[sub_idx] + torch.clamp(
                        new_value - flatten_values[sub_idx],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - flatten_returns[sub_idx]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_value - flatten_returns[sub_idx]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                self.agent.retropropagate(loss, max_grad_norm)

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

            y_pred, y_true = flatten_values.cpu().numpy(), flatten_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self.logger.add_scalar("charts/learning_rate",
                                   self.agent.optim.param_groups[0]["lr"],
                                   self.global_step)
            self.logger.add_scalar("losses/value_loss",
                                   v_loss.item(),
                                   self.global_step)
            self.logger.add_scalar("losses/policy_loss",
                                   pg_loss.item(),
                                   self.global_step)
            self.logger.add_scalar("losses/entropy_loss",
                                   entropy_loss.item(),
                                   self.global_step)
            self.logger.add_scalar("losses/old_approx_kl",
                                   old_approx_kl.item(),
                                   self.global_step)
            self.logger.add_scalar("losses/approx_kl",
                                   approx_kl.item(),
                                   self.global_step)
            self.logger.add_scalar("losses/clip_frac",
                                   np.mean(clip_fracs),
                                   self.global_step)
            self.logger.add_scalar("losses/explained_variance",
                                   explained_var,
                                   self.global_step)
            self.logger.add_scalar("charts/SPS",
                                   int(self.global_step / (time.time() - self.start_time)),
                                   self.global_step)

    def eval(self, debug=False):
        num_eval = 5
        eval_step = self.global_step

        for update in range(num_eval):
            eval_step += 1
            all_done = False
            all_info = dict((i, {}) for i in range(self.num_envs))

            while all_done is False:
                # rollouts (no need for gradient)
                with torch.no_grad():
                    action, logprob, entropy, crit_value = \
                        self.agent.get_action_and_critic(self.next_obs)

                # step the envs
                next_obs, reward, done, info = self.sims.step(action.cpu().numpy())

                for item, i in zip(info, all_info):
                    if "episode" not in all_info[i].keys() and "episode" in item.keys():
                        all_info[i] = item
                all_done = all([len(all_info[i].keys()) > 0 for i in all_info])

            avg_episodic_return = sum([all_info[i]['episode']['r'] for i in all_info]) / self.num_envs
            avf_episodic_length = sum([all_info[i]['episode']['l'] for i in all_info]) / self.num_envs
            self.logger.add_scalar("eval/episodic_return", avg_episodic_return, eval_step)
            self.logger.add_scalar("eval/episodic_length", avf_episodic_length, eval_step)
            self.logger.flush()

    def close(self):
        self.sims.close()
        self.logger.flush()
        self.logger.close()