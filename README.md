# Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization

[[Project website](https://clvrai.com/skill-chaining)] [[Paper](https://openreview.net/forum?id=K5-J-Espnaq)]

This project is a PyTorch implementation of [Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization](https://clvrai.com/skill-chaining), published in CoRL 2021.


### Note that Unity rendering for IKEA Furniture Assembly Environment is temporally not available due to the deprecated Unity-MuJoCo plugin in the new version of MuJoCo (2.1). It is still working with MuJoCo 2.0.


## Files and Directories
* `run.py`: launches an appropriate trainer based on algorithm
* `policy_sequencing_trainer.py`: trainer for policy sequencing
* `policy_sequencing_agent.py`: model and training code for policy sequencing
* `policy_sequencing_rollout.py`: rollout with policy sequencing agent
* `policy_sequencing_config.py`: hyperparameters
* `method/`: implementation of IL and RL algorithms
* `furniture/`: IKEA furniture environment
* `demos/`: default demonstration directory
* `log/`: default training log directory
* `result/`: evaluation result directory


## Prerequisites
* Ubuntu 18.04 or above
* Python 3.6
* Mujoco 2.1


## Installation

0. Clone this repository and submodules.
```bash
$ git clone --recursive git@github.com:clvrai/skill-chaining.git
```

1. Install mujoco 2.1 and add the following environment variables into `~/.bashrc` or `~/.zshrc`
Note that the code is compatible with **MuJoCo 2.0**, which supports Unity rendering.
```bash
# download mujoco 2.1
$ mkdir ~/.mujoco
$ wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco_linux.tar.gz
$ tar -xvzf mujoco_linux.tar.gz -C ~/.mujoco/
$ rm mujoco_linux.tar.gz

# add mujoco to LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# for GPU rendering
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# only for a headless server
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

2. Install python dependencies
```bash
$ sudo apt-get install cmake libopenmpi-dev libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libglew-dev

# software rendering
$ sudo apt-get install libgl1-mesa-glx libosmesa6 patchelf

# window rendering
$ sudo apt-get install libglfw3 libglew2.0
```

3. Install furniture submodule
```bash
$ cd furniture
$ pip install -e .
$ cd ../method
$ pip install -e .
$ pip install torch torchvision
```


## Usage

For `chair_ingolf_0650`, simply change `table_lack_0825` to `chair_ingolf_0650` in the commands. For training with gpu, specify the desired gpu number (e.g. `--gpu 0`). To change the random seed, append, e.g., `--seed 0` to the command.

To enable wandb logging, add the following arguments with your wandb entity and project names: `--wandb True --wandb_entity [WANDB ENTITY] --wandb_project [WANDB_PROJECT]`.


1. Generate demos
```
# Sub-task demo generation
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --demo_dir demos/table_lack/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --start_count 0 --phase_ob True
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --demo_dir demos/table_lack/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --preassembled 0 --start_count 1000 --phase_ob True
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --demo_dir demos/table_lack/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --preassembled 0,1 --start_count 2000 --phase_ob True
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --demo_dir demos/table_lack/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --preassembled 0,1,2 --start_count 3000 --phase_ob True

# Full-task demo generation
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --demo_dir demos/table_lack_full/ --reset_robot_after_attach True --max_episode_steps 800 --num_connects 4 --n_demos 200 --start_count 0 --phase_ob True
```

2. Train sub-task policies
```
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_0 --num_connects 1 --run_prefix p0
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_1 --num_connects 1 --preassembled 0 --run_prefix p1 --load_init_states log/table_lack_0825.gail.p0.123/success_00024576000.pkl
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_2 --num_connects 1 --preassembled 0,1 --run_prefix p2 --load_init_states log/table_lack_0825.gail.p1.123/success_00030310400.pkl
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_3 --num_connects 1 --preassembled 0,1,2 --run_prefix p3 --load_init_states log/table_lack_0825.gail.p2.123/success_00027852800.pkl
```

3. Collect successful terminal states from sub-task policies
Find the best performing checkpoint from WandB, and replace checkpoint path with the best performing checkpoint (e.g. `--init_ckpt_path log/table_lack_0825.gail.p0.123/ckpt_00021299200.pt`).
```
python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_0 --num_connects 1 --run_prefix p0 --is_train False --num_eval 200 --record_video False --init_ckpt_path log/table_lack_0825.gail.p0.123/ckpt_00000000000.pt
python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_1 --num_connects 1 --preassembled 0 --run_prefix p1 --is_train False --num_eval 200 --record_video False --init_ckpt_path log/table_lack_0825.gail.p1.123/ckpt_00000000000.pt
python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_2 --num_connects 1 --preassembled 0,1 --run_prefix p2 --is_train False --num_eval 200 --record_video False --init_ckpt_path log/table_lack_0825.gail.p2.123/ckpt_00000000000.pt
python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_3 --num_connects 1 --preassembled 0,1,2 --run_prefix p3 --is_train False --num_eval 200 --record_video False --init_ckpt_path log/table_lack_0825.gail.p3.123/ckpt_00000000000.pt
```

4. Train skill chaining
Use the best performing checkpoints (`--ps_ckpt`) and their successful terminal states (`--ps_laod_init_states`).
```
# Ours
mpirun -np 16 python -m run --algo ps --furniture_name table_lack_0825 --num_connects 4 --run_prefix ours \
--ps_ckpts log/table_lack_0825.gail.p0.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p1.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p2.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p3.123/ckpt_00021299200.pt \
--ps_load_init_states log/table_lack_0825.gail.p0.123/success_00021299200.pkl,log/table_lack_0825.gail.p1.123/success_00021299200.pkl,log/table_lack_0825.gail.p2.123/success_00021299200.pkl,log/table_lack_0825.gail.p3.123/success_00021299200.pkl \
--ps_demo_paths demos/table_lack/Sawyer_table_lack_0825_0,demos/table_lack/Sawyer_table_lack_0825_1,demos/table_lack/Sawyer_table_lack_0825_2,demos/table_lack/Sawyer_table_lack_0825_3

# Policy Sequencing (Clegg et al. 2018)
mpirun -np 16 python -m run --algo ps --furniture_name table_lack_0825 --num_connects 4 --run_prefix ps \
--ps_ckpts log/table_lack_0825.gail.p0.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p1.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p2.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p3.123/ckpt_00021299200.pt \
--ps_load_init_states log/table_lack_0825.gail.p0.123/success_00021299200.pkl,log/table_lack_0825.gail.p1.123/success_00021299200.pkl,log/table_lack_0825.gail.p2.123/success_00021299200.pkl,log/table_lack_0825.gail.p3.123/success_00021299200.pkl \
--ps_demo_paths demos/table_lack/Sawyer_table_lack_0825_0,demos/table_lack/Sawyer_table_lack_0825_1,demos/table_lack/Sawyer_table_lack_0825_2,demos/table_lack/Sawyer_table_lack_0825_3
```

5. Train baselines
```
# BC
python -m run --algo bc --max_global_step 1000 --furniture_name table_lack_0825 --demo_path demos/table_lack_full/Sawyer_table_lack_0825 --record_video False --run_prefix bc --gpu 0

# GAIL
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack_full/Sawyer_table_lack_0825 --num_connects 4 --max_episode_steps 800 --max_global_step 200000000 --run_prefix gail --gail_env_reward 0

# GAIL+PPO
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack_full/Sawyer_table_lack_0825 --num_connects 4 --max_episode_steps 800 --max_global_step 200000000 --run_prefix gail_ppo

# PPO
mpirun -np 16 python -m run --algo ppo --furniture_name table_lack_0825 --num_connects 4 --max_episode_steps 800 --max_global_step 200000000 --run_prefix ppo
```


## Citation
If you find this useful, please cite
```
@inproceedings{lee2021adversarial,
  title={Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization},
  author={Youngwoon Lee and Joseph J. Lim and Anima Anandkumar and Yuke Zhu},
  booktitle={Conference on Robot Learning},
  year={2021},
}
```


## References
- This code is based on Youngwoon's robot-learning repo: https://github.com/youngwoon/robot-learning
- IKEA Furniture Assembly Environment: https://github.com/clvrai/furniture
