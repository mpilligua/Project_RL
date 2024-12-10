NUMBER_OF_REWARDS_TO_AVERAGE = 100          

# ----------------- Prioritized Buffer Configs ----------------- #

prioritized_buffer_config = {
    "burn_in": 10000,
    "alpha": 0.6,
    "small_constant": 0.05,
    "growth_rate": 1.0005,
    "beta": 0.4
}

# ----------------- Training Configs ----------------- #
dqn_extensions = {
    "use_noisy_dqn": False,
    "use_double_dqn": False,
    "use_two_step_dqn": False,
    "use_dueling_dqn": False,
    "use_prioritized_buffer": False
}

training_config = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_decay": 0.999985,
    "eps_min": 0.02,
    "dnn_upd": 5,
    "dnn_sync": 1000,
    "noise_upd": 11000,
    "max_frames": 1000000,
    "dqn_extensions": dqn_extensions,
    "experience_replay_size": 10000,
    "prioritized_buffer_config": prioritized_buffer_config
}

# ----------------- Env Configs ----------------- #

train_env_config = {
    "env_name": "ALE/Tennis-v5",
    "skip": 4,
    "stack_size": 4,
    "reshape_size": (84, 84),
    "render_mode": None
}

eval_env_config = {
    "env_name": "ALE/Tennis-v5",
    "skip": 1,
    "stack_size": 4,
    "reshape_size": (84, 84),
    "render_mode": 'rgb_array',
    "eval": True,
}


all_configs = {
    "training_config": training_config,
    "train_env_config": train_env_config,
    "eval_env_config": eval_env_config,
}