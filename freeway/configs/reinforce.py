NUMBER_OF_REWARDS_TO_AVERAGE = 100          

# ----------------- Prioritized Buffer Configs ----------------- #


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
    "experience_replay_size": 10000,
    "burn_in": 10000,
    "steps_scheduler": 10,
}

# ----------------- Env Configs ----------------- #

train_env_config = {
    "env_name": "ALE/Freeway-v5",
    "skip": 4,
    "stack_size": 4,
    "reshape_size": (84, 84),
    "render_mode": None
}

eval_env_config = {
    "env_name": "ALE/Freeway-v5",
    "skip": 4,
    "stack_size": 4,
    "reshape_size": (84, 84),
    "render_mode": 'rgb_array'
}


all_configs = {
    "training_config": training_config,
    "train_env_config": train_env_config,
    "eval_env_config": eval_env_config,
}