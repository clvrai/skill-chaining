"""
Collects policy sequencing rollouts (PolicySequencingRolloutRunner class).
"""

import pickle
import os

import numpy as np
from mujoco_py.builder import MujocoException

from robot_learning.algorithms.rollout import Rollout, RolloutRunner
from robot_learning.utils.logger import logger
from robot_learning.utils.info_dict import Info
from robot_learning.utils.gym_env import get_non_absorbing_state, zero_value


class PolicySequencingRolloutRunner(RolloutRunner):
    """
    Run rollout given environment and multiple sub-policies.
    """

    def __init__(self, config, env, env_eval, agent):
        """
        Args:
            config: configurations for the environment.
            env: training environment.
            env_eval: testing environment.
            agent: policy.
        """
        super().__init__(config, env, env_eval, agent)

        self._n_subtask = env.num_subtask()
        self._subtask = 0
        self._init_sampling = 0

        # initialize env with stored states
        if config.ps_load_init_states and config.is_train:
            for i, path in enumerate(config.ps_load_init_states):
                if path and i + 1 < self._n_subtask:
                    path = os.path.expanduser(path)
                    with open(path, "rb") as f:
                        states = pickle.load(f)
                        self._agent.initial_states[i + 1].extend(states)
                        self._agent.initial_state_dists[i + 1].update(states)

    def _reset_env(self, env, subtask=None, num_connects=None):
        """ Resets the environment and return the initial observation. """
        if subtask is None:
            subtask = self._subtask
        env.set_subtask(subtask, num_connects)
        p = np.random.rand()
        init_qpos = None
        self._init_sampling = 0

        if subtask > 0 and len(self._agent.initial_states[subtask]) > 0:
            if p < self._config.ps_env_init_from_dist:
                init_qpos = self._agent.initial_state_dists[subtask].sample(1)
                self._init_sampling = 1
            elif p > 1 - self._config.ps_env_init_from_states:
                init_qpos = np.random.choice(self._agent.initial_states[subtask])
                self._init_sampling = 2

        env.set_init_qpos(init_qpos)
        try:
            ret = env.reset()
        except MujocoException:
            logger.error("Fail to initialize env with %s", init_qpos)
            env.set_init_qpos(None)
            ret = env.reset()

        return ret

    def switch_subtask(self, subtask):
        self._subtask = subtask

    def run(
        self,
        is_train=True,
        every_steps=None,
        every_episodes=None,
        log_prefix="",
        step=0,
    ):
        """
        Collects trajectories and yield every @every_steps/@every_episodes.

        Args:
            is_train: whether rollout is for training or evaluation.
            every_steps: if not None, returns rollouts @every_steps
            every_episodes: if not None, returns rollouts @every_epiosdes
            log_prefix: log as @log_prefix rollout: %s
        """
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")

        config = self._config
        env = self._env if is_train else self._env_eval
        agent = self._agent
        il = hasattr(agent[0], "predict_reward")

        # initialize rollout buffer
        rollout = Rollout()
        reward_info = Info()
        ep_info = Info()
        episode = 0

        while True:
            done = False
            ep_len = 0
            ep_rew = 0
            ep_rew_rl = 0
            if il:
                ep_rew_il = 0

            ob_init = ob_next = self._reset_env(env, num_connects=1)
            subtask = self._subtask
            next_subtask = subtask + 1
            reward_tstar = 0

            # run rollout
            while not done:
                ob = ob_next

                # sample action from policy
                ac, ac_before_activation = agent[subtask].act(ob, is_train=is_train)

                # take a step
                ob_next, reward, done, info = env.step(ac)

                # if subtask succeeds
                if "subtask" in info and subtask != info["subtask"]:
                    agent.initial_obs[subtask].append(ob_init)
                    agent.terminal_obs[subtask].append(ob_next)
                    ob_init = ob_next

                    # add termination state regularization reward
                    if next_subtask < self._n_subtask and config.ps_use_tstar:
                        reward_tstar = agent.predict_tstar_reward(ob, next_subtask)
                        reward += config.ps_tstar_reward * reward_tstar

                if il:
                    reward_il = agent[subtask].predict_reward(ob, ob_next, ac)
                    reward_rl = (
                        (1 - config.gail_env_reward) * reward_il
                        + config.gail_env_reward * reward * config.reward_scale
                    )
                else:
                    reward_rl = reward * config.reward_scale

                step += 1
                ep_len += 1
                ep_rew += reward
                ep_rew_rl += reward_rl
                if il:
                    ep_rew_il += reward_il

                if done and ep_len < env.max_episode_steps:
                    done_mask = 0  # -1 absorbing, 0 done, 1 not done
                else:
                    done_mask = 1

                rollout.add(
                    {
                        "ob": ob,
                        "ob_next": ob_next,
                        "ac": ac,
                        "ac_before_activation": ac_before_activation,
                        "done": done,
                        "rew": reward,
                        "done_mask": done_mask,  # -1 absorbing, 0 done, 1 not done
                    }
                )

                reward_info.add(info)

                if config.absorbing_state and done_mask == 0:
                    absorbing_state = env.get_absorbing_state()
                    absorbing_action = zero_value(env.action_space)
                    rollout._history["ob_next"][-1] = absorbing_state
                    rollout.add(
                        {
                            "ob": absorbing_state,
                            "ob_next": absorbing_state,
                            "ac": absorbing_action,
                            "ac_before_activation": absorbing_action,
                            "rew": 0.0,
                            "done": 0,
                            "done_mask": -1,  # -1 absorbing, 0 done, 1 not done
                        }
                    )

                if every_steps is not None and step % every_steps == 0:
                    yield rollout.get(), ep_info.get_dict(only_scalar=True)

            # add successful final states to the next subtask's initial states
            if (
                config.is_train
                and config.ps_use_terminal_states
                and "episode_success_state" in reward_info.keys()
                and (self._init_sampling > 0 or subtask == 0)
                and next_subtask < self._n_subtask
            ):
                state = reward_info["episode_success_state"]
                self._agent.initial_states[next_subtask].extend(state)
                self._agent.initial_state_dists[next_subtask].update(state)

            # compute average/sum of information
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            reward_info_dict.update({"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl})
            if il:
                reward_info_dict["rew_il"] = ep_rew_il
            reward_info_dict["rew_tstar"] = reward_tstar
            ep_info.add(reward_info_dict)

            logger.info(
                log_prefix + " rollout: %s",
                {
                    k: v
                    for k, v in reward_info_dict.items()
                    if k not in self._exclude_rollout_log and np.isscalar(v)
                },
            )

            episode += 1
            if every_episodes is not None and episode % every_episodes == 0:
                yield rollout.get(), ep_info.get_dict(only_scalar=True)

    def run_episode(self, is_train=True, record_video=False, partial=False):
        """
        Runs one episode and returns the rollout (mainly for evaluation).

        Args:
            is_train: whether rollout is for training or evaluation.
            record_video: record video of rollout if True.
            partial: run each subtask policy.
        """
        config = self._config
        env = self._env if is_train else self._env_eval
        agent = self._agent
        il = hasattr(agent[0], "predict_reward")

        # initialize rollout buffer
        rollout = Rollout()
        reward_info = Info()
        if partial:
            subtask = self._subtask
            num_connects = 1
        else:
            subtask = 0
            num_connects = None
            env.set_max_episode_steps(config.max_episode_steps * 2)

        done = False
        ep_len = 0
        ep_rew = 0
        ep_rew_rl = 0
        if il:
            ep_rew_il = 0
        reward_tstar = 0

        ob_next = self._reset_env(env, subtask, num_connects)

        record_frames = []
        if record_video:
            record_frames.append(self._store_frame(env, ep_len, ep_rew))

        # run rollout
        while not done:
            ob = ob_next

            # sample action from policy
            ac, ac_before_activation = agent[subtask].act(ob, is_train=is_train)

            # take a step
            ob_next, reward, done, info = env.step(ac)
            if il:
                reward_il = agent[subtask].predict_reward(ob, ob_next, ac)

            next_subtask = subtask + 1
            if "subtask" in info and subtask != info["subtask"]:
                subtask = info["subtask"]

                # replace reward
                if next_subtask < self._n_subtask and config.ps_use_tstar:
                    reward_tstar = agent.predict_tstar_reward(ob, next_subtask)
                    reward += config.ps_tstar_reward * reward_tstar

            if il:
                reward_rl = (
                    (1 - config.gail_env_reward) * reward_il
                    + config.gail_env_reward * reward * config.reward_scale
                )
            else:
                reward_rl = reward * config.reward_scale

            ep_len += 1
            ep_rew += reward
            ep_rew_rl += reward_rl
            if il:
                ep_rew_il += reward_il

            rollout.add(
                {
                    "ob": ob,
                    "ac": ac,
                    "ac_before_activation": ac_before_activation,
                    "done": done,
                    "rew": reward,
                }
            )

            reward_info.add(info)
            if record_video:
                frame_info = info.copy()
                if il:
                    frame_info.update(
                        {
                            "ep_rew_il": ep_rew_il,
                            "rew_il": reward_il,
                            "rew_rl": reward_rl,
                            "rew_tstar": reward_tstar,
                        }
                    )
                record_frames.append(self._store_frame(env, ep_len, ep_rew, frame_info))

        # add last observation
        rollout.add({"ob": ob_next})

        # compute average/sum of information
        ep_info = {"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl}
        if il:
            ep_info["rew_il"] = ep_rew_il
        ep_info["rew_tstar"] = reward_tstar
        if "episode_success_state" in reward_info.keys():
            ep_info["episode_success_state"] = reward_info["episode_success_state"]
        ep_info.update(reward_info.get_dict(reduction="sum", only_scalar=True))

        logger.info(
            "rollout: %s",
            {
                k: v
                for k, v in ep_info.items()
                if k not in self._exclude_rollout_log and np.isscalar(v)
            },
        )

        if not partial:
            env.set_max_episode_steps(config.max_episode_steps)

        return rollout.get(), ep_info, record_frames
