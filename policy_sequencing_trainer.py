"""
Policy Sequencing and T-STAR training.
"""

import os
import pickle
import gc
from time import time

import torch
import wandb
import h5py
import numpy as np
from tqdm import tqdm, trange

from robot_learning.trainer import Trainer
from robot_learning.utils.info_dict import Info
from robot_learning.utils.logger import logger
from robot_learning.utils.mpi import mpi_sum, mpi_gather_average

from policy_sequencing_agent import PolicySequencingAgent
from policy_sequencing_rollout import PolicySequencingRolloutRunner


class PolicySequencingTrainer(Trainer):
    """
    Trainer class for Policy Sequencing and T-STAR in PyTorch.
    """

    def train(self):
        """ Trains a policy sequencing agent. """
        config = self._config

        # load checkpoint
        ckpt_info = self._load_ckpt(config.init_ckpt_path, config.ckpt_num)
        step = ckpt_info.get("step", 0)
        ps_epoch = ckpt_info.get("ps_epoch", 0)
        update_iter = ckpt_info.get("update_iter", 0)

        num_agent = self._agent._num_agents
        max_global_step = (
            config.ps_sub_policy_update_steps * num_agent * config.ps_epochs
        )

        # sync the networks across the cpus
        self._agent.sync_networks()

        logger.warn("Start training at step=%d", step)
        if self._is_chef:
            pbar = tqdm(initial=step, total=max_global_step, desc=config.run_name)
            ep_info = Info()
            train_info = Info()

        st_time = time()
        st_step = step

        while ps_epoch < config.ps_epochs:
            # train sub-policies one-by-one
            for i in range(num_agent):
                logger.warn("Epoch=%d   Agent=%d", ps_epoch, i)

                step_i = 0
                self._runner.switch_subtask(i)
                self._agent.initial_state_dists[i].recompute_stats()

                if self._is_chef:
                    ep_info = Info()
                    train_info = Info()

                # decide how many episodes or how long rollout to collect
                runner = self._runner.run(
                    every_steps=config.rollout_length, log_prefix=f"subtask {i}"
                )

                while step_i < config.ps_sub_policy_update_steps:
                    # collect rollouts
                    rollout, info = next(runner)
                    info = mpi_gather_average(info)
                    self._agent.store_episode(rollout, i)
                    step_per_batch = mpi_sum(len(rollout["ac"]))
                    logger.warn("Data collected")

                    # train an agent
                    _train_info = self._agent.train(i)
                    logger.info("Networks trained")

                    if runner and step < config.max_ob_norm_step:
                        self._update_normalizer(rollout, i)
                        logger.info("Normalizer updated")

                    step += step_per_batch
                    step_i += step_per_batch
                    update_iter += 1

                    # log training and episode information or evaluate
                    if self._is_chef:
                        pbar.update(step_per_batch)
                        ep_info.add(info)
                        train_info.add(_train_info)

                        if update_iter % config.log_interval == 0:
                            train_info.add(
                                {
                                    "training_agent": i,
                                    "sec": (time() - st_time) / config.log_interval,
                                    "steps_per_sec": (step - st_step)
                                    / (time() - st_time),
                                    "update_iter": update_iter,
                                }
                            )
                            st_time = time()
                            st_step = step
                            self._log_train(
                                step,
                                train_info.get_dict(),
                                ep_info.get_dict(),
                                name="_" + str(i),
                            )
                            ep_info = Info()
                            train_info = Info()

                        if update_iter % config.evaluate_interval == 1:
                            logger.info("Evaluate at %d", update_iter)
                            rollout, info = self._evaluate_partial(
                                step=step,
                                record_video=config.record_video,
                                partial=True,
                            )
                            self._log_test(step, info, name="_" + str(i))
                            rollout, info = self._evaluate_partial(
                                step=step,
                                record_video=config.record_video,
                                partial=False,
                            )
                            self._log_test(step, info)
                            logger.warn("Garbage collection: %s", str(gc.get_count()))

                        if update_iter % config.ckpt_interval == 0:
                            self._save_ckpt(
                                step, {"ps_epoch": ps_epoch, "update_iter": update_iter}
                            )
            ps_epoch += 1

        if self._is_chef:
            self._save_ckpt(step, {"ps_epoch": ps_epoch, "update_iter": update_iter})

        logger.info("Reached %s steps. worker %d stopped.", step, config.rank)

    def evaluate(self):
        """ Evaluates an agent stored in chekpoint with @self._config.ckpt_num. """
        ckpt_info = self._load_ckpt(self._config.init_ckpt_path, self._config.ckpt_num)
        step = ckpt_info.get("step", 0)
        ps_epoch = ckpt_info.get("ps_epoch", 0)
        update_iter = ckpt_info.get("update_iter", 0)

        logger.info(
            "Run %d evaluations at step=%d, ps_epoch=%d, update_iter=%d",
            self._config.num_eval,
            step,
            ps_epoch,
            update_iter,
        )

        # for i in range(self._agent._num_agents):
        #     self._runner.switch_subtask(i)
        #     rollouts, info = self._evaluate_partial(
        #         step=step, record_video=self._config.record_video, partial=True
        #     )

        rollouts, info = self._evaluate_partial(
            step=step, record_video=self._config.record_video, partial=False
        )
        logger.info("Done evaluating %d episodes", self._config.num_eval)

        if "episode_success_state" in info.keys():
            success_states = info["episode_success_state"]
            fname = "success_{:011d}.pkl".format(step)
            path = os.path.join(self._config.log_dir, fname)
            logger.warn(
                "[*] Store {} successful terminal states: {}".format(
                    len(success_states), path
                )
            )
            with open(path, "wb") as f:
                pickle.dump(success_states, f)

        info_stat = info.get_stat()
        os.makedirs("result", exist_ok=True)
        with h5py.File("result/{}.hdf5".format(self._config.run_name), "w") as hf:
            for k, v in info.items():
                if np.isscalar(v) or isinstance(
                    v[0], (int, float, bool, np.float32, np.int64, np.ndarray)
                ):
                    hf.create_dataset(k, data=v)
        with open("result/{}.txt".format(self._config.run_name), "w") as f:
            for k, v in info_stat.items():
                f.write("{}\t{:.03f} $\\pm$ {:.03f}\n".format(k, v[0], v[1]))

        if self._config.record_demo:
            new_rollouts = []
            for rollout in rollouts:
                new_rollout = {
                    "obs": rollout["ob"],
                    "actions": rollout["ac"],
                    "rewards": rollout["rew"],
                    "dones": rollout["done"],
                }
                new_rollouts.append(new_rollout)

            fname = "{}_step_{:011d}_{}_trajs.pkl".format(
                self._config.run_name,
                step,
                self._config.num_eval,
            )
            path = os.path.join(self._config.demo_dir, fname)
            logger.warn("[*] Generating demo: {}".format(path))
            with open(path, "wb") as f:
                pickle.dump(new_rollouts, f)

    def _get_agent_by_name(self, algo):
        """ Returns RL or IL agent. """
        if algo == "ps":
            return PolicySequencingAgent
        else:
            return super()._get_agent_by_name(algo)

    def _get_runner_by_name(self, algo):
        """ Returns rollout runner for @algo. """
        if algo == "ps":
            return PolicySequencingRolloutRunner
        else:
            return super()._get_runner_by_name(algo)

    def _update_normalizer(self, rollout, i):
        """ Updates normalizer with @rollout. """
        if self._config.ob_norm:
            self._agent.update_normalizer(rollout["ob"], i)

    def _evaluate_partial(self, step=None, record_video=False, partial=True):
        """
        Runs one rollout if in eval mode (@idx is not None).
        Runs num_record_samples rollouts if in train mode (@idx is None).

        Args:
            step: the number of environment steps.
            record_video: whether to record video or not.
            partial: evaluate each subtask policy.
        """
        logger.info("Run %d evaluations at step=%d", self._config.num_eval, step)
        rollouts = []
        info_history = Info()
        for i in range(self._config.num_eval):
            logger.warn("Evalute run %d", i + 1)
            rollout, info, frames = self._runner.run_episode(
                is_train=False, record_video=record_video, partial=partial
            )
            rollouts.append(rollout)

            if record_video:
                ep_rew = info["rew"]
                ep_success = (
                    "s"
                    if "episode_success" in info and info["episode_success"]
                    else "f"
                )
                fname = "{}_step_{:011d}_{}_r_{:.3f}_{}{}.mp4".format(
                    self._config.env,
                    step,
                    i,
                    ep_rew,
                    "partial_" if partial else "",
                    ep_success,
                )
                video_path = self._save_video(fname, frames)
                if self._config.is_train:
                    caption = "{}-{}-{}".format(self._config.run_name, step, i)
                    if partial:
                        caption += "-partial"
                    info["video"] = wandb.Video(
                        video_path, caption=caption, fps=15, format="mp4"
                    )

            info_history.add(info)

        return rollouts, info_history
