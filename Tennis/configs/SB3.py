NUMBER_OF_REWARDS_TO_AVERAGE = 100          

# ----------------- Model Configs ----------------- #
dqn_arguments = {
    "policy_type": "CnnPolicy",
    "buffer_size": 30000,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "learning_starts": 100000,
    "target_update_interval": 1000,
    "train_freq": 4,
    "gradient_steps": 1,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.01
}

ppo_arguments = {
    "model_name": "ppo",
    "policy_type": "MultiInputPolicy",
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "rollout_buffer_class": None,
    "rollout_buffer_kwargs": None,
    "target_kl": None,
    "stats_window_size": 100,
    "tensorboard_log": None,
    "policy_kwargs": None,
    "verbose": 0,
    "seed": None,
    "device": "auto",
    "_init_setup_model": True
}


ac2_arguments = {
    "model_name": "a2c",
    "policy_type": "CnnPolicy",
    "learning_rate": 0.0003}

sac_arguments = {
    "model_name": "sac",
    "policy_type": "CnnPolicy",
}

list_of_wrappers = {"redChannel": False,
                    "clip_reward": False,
                    "VecFrameStack": False,
                    "ScaledFloatFrame":False,
                    "GetBallPosition": True,
                    "VecTransposeImage": True,
                    "ReshapeObservation": False,
                    "ResizeObservation": True,
                    "CropObservation": True
                    }


training_config = {"total_timesteps": 40000000, 
                  "model_name": 'a2c',
                   "n_envs": 1,
                   "eval_freq": 10,
                   "custom_dqn": False,
                   "number_of_actions": "All",
                   "training_config": True}

# ----------------- Env Configs ----------------- #
train_env_config = {
    "env_name": "ALE/Tennis-v5",
    "skip": 4,
    "stack_size": 4,
    "reshape_size": (84, 84),
    "render_mode": 'rgb_array',
}

eval_env_config = {
    "env_name": "ALE/Tennis-v5",
    "skip": 1,
    "stack_size": 4,
    "reshape_size": (84, 84),
    "render_mode": 'rgb_array',
    "eval": 'rgb_array',
}

all_configs = {
    "training_config": training_config,
    "train_env_config": train_env_config,
    "eval_env_config": eval_env_config,
}
