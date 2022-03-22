from collections import OrderedDict
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.distributions
from torch.optim.lr_scheduler import StepLR

from robot_learning.algorithms.base_agent import BaseAgent
from robot_learning.algorithms.gail_agent import GAILAgent
from robot_learning.algorithms.ppo_agent import PPOAgent
from robot_learning.algorithms.expert_dataset import ExpertDataset
from robot_learning.networks.discriminator import Discriminator
from robot_learning.utils.normalizer import Normalizer
from robot_learning.utils.info_dict import Info
from robot_learning.utils.logger import logger
from robot_learning.utils.mpi import mpi_average
from robot_learning.utils.gym_env import value_to_space
from robot_learning.utils.pytorch import (
    get_ckpt_path,
    optimizer_cuda,
    count_parameters,
    sync_networks,
    sync_grads,
    to_tensor,
    obs2tensor,
)


class PolicySequencingAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._num_agents = len(config.ps_ckpts)
        self._rl_agents = []
        for i in range(self._num_agents):
            config_i = copy.copy(config)
            config_i.demo_path = config.ps_demo_paths[i]
            if config.ps_rl_algo == "gail":
                self._rl_agents.append(
                    GAILAgent(config_i, ob_space, ac_space, env_ob_space)
                )
            elif config.ps_rl_algo == "ppo":
                self._rl_agents.append(
                    PPOAgent(config_i, ob_space, ac_space, env_ob_space)
                )
                self._rl_agents[i]._dataset = ExpertDataset(
                    config_i.demo_path,
                    config.demo_subsample_interval,
                    ac_space,
                    use_low_level=config.demo_low_level,
                    sample_range_start=config.demo_sample_range_start,
                    sample_range_end=config.demo_sample_range_end,
                )

        for i, ckpt in enumerate(config.ps_ckpts):
            ckpt_path, ckpt_num = get_ckpt_path(ckpt, ckpt_num=None)
            assert ckpt_path is not None, "Cannot find checkpoint at %s" % ckpt_dir

            logger.warn("Load checkpoint %s", ckpt_path)
            self._rl_agents[i].load_state_dict(
                torch.load(ckpt_path, map_location=self._config.device)["agent"]
            )

        if config.ps_use_tstar:
            self._discriminators = []
            if config.ps_discriminator_loss_type == "gan":
                self._discriminator_loss = nn.BCEWithLogitsLoss()
            elif config.ps_discriminator_loss_type == "lsgan":
                self._discriminator_loss = nn.MSELoss()
            for i in range(self._num_agents):
                self._discriminators.append(
                    Discriminator(
                        config,
                        ob_space,
                        mlp_dim=config.ps_discriminator_mlp_dim,
                        activation=config.ps_discriminator_activation,
                    )
                )
            self._network_cuda(config.device)

        if config.is_train and config.ps_use_tstar:
            self._discriminator_optims = []
            self._discriminator_lr_schedulers = []
            for i in range(self._num_agents):
                # build optimizers
                self._discriminator_optims.append(
                    optim.Adam(
                        self._discriminators[i].parameters(),
                        lr=config.ps_discriminator_lr,
                    )
                )

                # build learning rate scheduler
                self._discriminator_lr_schedulers.append(
                    StepLR(
                        self._discriminator_optims[i],
                        step_size=self._config.max_global_step
                        // self._config.rollout_length,
                        gamma=0.5,
                    )
                )

        # expert dataset
        self.initial_states = []  # for environment init state
        self.initial_state_dists = []  # for environment init state distribution
        self.initial_obs = []  # for constraining terminal state
        self.terminal_obs = []  # for constraining terminal state
        for i in range(self._num_agents):
            self.initial_states.append(self._rl_agents[i]._dataset.initial_states)
            self.initial_obs.append(self._rl_agents[i]._dataset.initial_obs)
            self.terminal_obs.append(self._rl_agents[i]._dataset.terminal_obs)
            state_shape = value_to_space(self.initial_states[i][0])
            self.initial_state_dists.append(Normalizer(state_shape, eps=0))
            self.initial_state_dists[i].update(self.initial_states[i])
            self.initial_state_dists[i].recompute_stats()

        self._log_creation()

    def __getitem__(self, key):
        return self._rl_agents[key]

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a policy sequencing agent")

    def is_off_policy(self):
        return False

    def store_episode(self, rollouts, agent_idx):
        self._rl_agents[agent_idx].store_episode(rollouts)

    def state_dict(self):
        ret = {
            "rl_agents": [agent.state_dict() for agent in self._rl_agents],
            "initial_states": self.initial_states,
            "initial_obs": self.initial_obs,
            "terminal_obs": self.terminal_obs,
            "initial_state_dists": [
                dist.state_dict() for dist in self.initial_state_dists
            ],
        }
        if self._config.ps_use_tstar:
            ret["discriminators_state_dict"] = [
                d.state_dict() for d in self._discriminators
            ]
            ret["discriminator_optims_state_dict"] = [
                o.state_dict() for o in self._discriminator_optims
            ]

        return ret

    def load_state_dict(self, ckpt):
        for i in range(self._num_agents):
            self._rl_agents[i].load_state_dict(ckpt["rl_agents"][i])
            if self._config.ps_use_tstar:
                self._discriminators[i].load_state_dict(
                    ckpt["discriminators_state_dict"][i]
                )
                if self._config.is_train:
                    self._discriminator_optims[i].load_state_dict(
                        ckpt["discriminator_optims_state_dict"][i]
                    )
                    optimizer_cuda(self._discriminator_optims[i], self._config.device)
            if self._config.is_train:
                self.initial_state_dists[i].load_state_dict(
                    ckpt["initial_state_dists"][i]
                )
        if self._config.is_train:
            self.initial_states = ckpt["initial_states"]
            self.initial_obs = ckpt["initial_obs"]
            self.terminal_obs = ckpt["terminal_obs"]
        self._network_cuda(self._config.device)

    def _network_cuda(self, device):
        if self._config.ps_use_tstar:
            for i in range(self._num_agents):
                self._discriminators[i].to(device)

    def sync_networks(self):
        for i in range(self._num_agents):
            self._rl_agents[i].sync_networks()
            if self._config.ps_use_tstar:
                sync_networks(self._discriminators[i])

    def update_normalizer(self, obs=None, i=None):
        if obs is not None and i is not None:
            self._rl_agents[i].update_normalizer(obs)

    def _predict_tstar_reward(self, ob, agent_idx):
        d = self._discriminators[agent_idx]
        d.eval()
        with torch.no_grad():
            ret = d(ob)
            eps = 1e-10
            s = torch.sigmoid(ret)
            if self._config.ps_tstar_reward_type == "vanilla":
                reward = -(1 - s + eps).log()
            elif self._config.ps_tstar_reward_type == "gan":
                reward = (s + eps).log() - (1 - s + eps).log()
            elif self._config.ps_tstar_reward_type == "d":
                reward = ret
            elif self._config.ps_tstar_reward_type == "amp":
                ret = torch.clamp(ret, 0, 1) - 1
                reward = 1 - ret ** 2
        d.train()
        return reward

    def predict_tstar_reward(self, ob, agent_idx):
        ob = self.normalize(ob)
        ob = to_tensor(ob, self._config.device)
        reward = self._predict_tstar_reward(ob, agent_idx)
        return reward.cpu().item()

    def train(self, agent_idx):
        train_info = Info()

        if self._config.ps_use_tstar and agent_idx > 0:
            num_batches = (
                self._config.rollout_length
                // self._config.batch_size
                // self._config.ps_discriminator_update_freq
            )
            assert num_batches > 0

            self._discriminator_lr_schedulers[agent_idx].step()

            expert_dataset = self.initial_obs[agent_idx]
            policy_dataset = self.terminal_obs[agent_idx - 1]
            for _ in range(num_batches):
                # policy_data = self._rl_agents[agent_idx]._buffer.sample(
                #     self._config.batch_size
                # )
                idxs = np.random.randint(
                    0, len(policy_dataset), self._config.batch_size
                )
                states = [policy_dataset[idx] for idx in idxs]
                if isinstance(states[0], dict):
                    sub_keys = states[0].keys()
                    new_states = {
                        sub_key: np.stack([v[sub_key] for v in states])
                        for sub_key in sub_keys
                    }
                else:
                    new_states = np.stack(states)
                policy_data = {"ob": new_states}

                idxs = np.random.randint(
                    0, len(expert_dataset), self._config.batch_size
                )
                states = [expert_dataset[idx] for idx in idxs]
                if isinstance(states[0], dict):
                    sub_keys = states[0].keys()
                    new_states = {
                        sub_key: np.stack([v[sub_key] for v in states])
                        for sub_key in sub_keys
                    }
                else:
                    new_states = np.stack(states)
                expert_data = {"ob": new_states}

                _train_info = self._update_discriminator(
                    agent_idx, policy_data, expert_data
                )
                train_info.add(_train_info)

        _train_info = self._rl_agents[agent_idx].train()
        train_info.add(_train_info)

        # ob normalization?

        return train_info.get_dict(only_scalar=True)

    def _update_discriminator(self, i, policy_data, expert_data):
        info = Info()

        _to_tensor = lambda x: to_tensor(x, self._config.device)
        # pre-process observations
        p_o = policy_data["ob"]
        p_o = self.normalize(p_o)
        p_o = _to_tensor(p_o)

        e_o = expert_data["ob"]
        e_o = self.normalize(e_o)
        e_o = _to_tensor(e_o)

        p_logit = self._discriminators[i](p_o)
        e_logit = self._discriminators[i](e_o)

        if self._config.ps_discriminator_loss_type == "lsgan":
            p_output = p_logit
            e_output = e_logit
        else:
            p_output = torch.sigmoid(p_logit)
            e_output = torch.sigmoid(e_logit)

        p_loss = self._discriminator_loss(
            p_logit, torch.zeros_like(p_logit).to(self._config.device)
        )
        e_loss = self._discriminator_loss(
            e_logit, torch.ones_like(e_logit).to(self._config.device)
        )

        logits = torch.cat([p_logit, e_logit], dim=0)
        entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean()
        entropy_loss = -self._config.ps_entropy_loss_coeff * entropy

        grad_pen = self._compute_grad_pen(i, p_o, e_o)
        grad_pen_loss = self._config.ps_grad_penalty_coeff * grad_pen

        ps_loss = p_loss + e_loss + entropy_loss + grad_pen_loss

        # update the discriminator
        self._discriminators[i].zero_grad()
        ps_loss.backward()
        sync_grads(self._discriminators[i])
        self._discriminator_optims[i].step()

        info["ps_disc_policy_output"] = p_output.mean().detach().cpu().item()
        info["ps_disc_expert_output"] = e_output.mean().detach().cpu().item()
        info["ps_disc_entropy"] = entropy.detach().cpu().item()
        info["ps_disc_policy_loss"] = p_loss.detach().cpu().item()
        info["ps_disc_expert_loss"] = e_loss.detach().cpu().item()
        info["ps_disc_entropy_loss"] = entropy_loss.detach().cpu().item()
        info["ps_disc_grad_pen"] = grad_pen.detach().cpu().item()
        info["ps_disc_grad_loss"] = grad_pen_loss.detach().cpu().item()
        info["ps_disc_loss"] = ps_loss.detach().cpu().item()

        return mpi_average(info.get_dict(only_scalar=True))

    def _compute_grad_pen(self, i, policy_ob, expert_ob):
        batch_size = self._config.batch_size
        alpha = torch.rand(batch_size, 1, device=self._config.device)

        def blend_dict(a, b, alpha):
            if isinstance(a, dict):
                return OrderedDict(
                    [(k, blend_dict(a[k], b[k], alpha)) for k in a.keys()]
                )
            elif isinstance(a, list):
                return [blend_dict(a[i], b[i], alpha) for i in range(len(a))]
            else:
                expanded_alpha = alpha.expand_as(a)
                ret = expanded_alpha * a + (1 - expanded_alpha) * b
                ret.requires_grad = True
                return ret

        interpolated_ob = blend_dict(policy_ob, expert_ob, alpha)
        inputs = list(interpolated_ob.values())

        interpolated_logit = self._discriminators[i](interpolated_ob)
        ones = torch.ones(interpolated_logit.size(), device=self._config.device)

        grad = autograd.grad(
            outputs=interpolated_logit,
            inputs=inputs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
