import os
from datetime import datetime
from stable_baselines3 import DQN, A2C, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, ReshapeObservation, FrameStackObservation
import os
import ale_py

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Custom Observation Wrappers
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


from stable_baselines3.common.callbacks import BaseCallback

class VideoRecorderCallback(BaseCallback):
    def __init__(self, video_folder, video_freq=1000, video_length=1000, verbose=0):
        super().__init__(verbose)
        self.video_folder = video_folder
        self.video_freq = video_freq
        self.video_length = video_length
        self.last_video_step = 0

    def _on_step(self) -> bool:
        print("doind someitn")
        if self.n_calls - self.last_video_step >= self.video_freq:
            self.last_video_step = self.n_calls
            video_name = f"agent-training-step-{self.n_calls}-to-step-{self.n_calls + self.video_length}.mp4"
            video_path = os.path.join(self.video_folder, video_name)
            
            # Record the video
            vec_env = VecVideoRecorder(
                self.training_env,
                self.video_folder,
                record_video_trigger=lambda _: True,
                video_length=self.video_length,
                name_prefix=f"agent-training-step-{self.n_calls}"
            )
            vec_env.reset()
            
            for _ in range(self.video_length):
                action, _ = self.model.predict(self.training_env.render(mode="rgb_array"))
                vec_env.step(action)

            vec_env.close()
            if self.verbose > 0:
                print(f"Saved video: {video_path}")
        return True


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


# Environment creation function
def make_env(env_name, skip=4, stack_size=4, reshape_size=(84, 84), render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = MaxAndSkipObservation(env, skip=skip)
    env = ResizeObservation(env, reshape_size)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ImageToPyTorch(env)
    env = ReshapeObservation(env, reshape_size)
    env = FrameStackObservation(env, stack_size=stack_size)
    env = ScaledFloatFrame(env)
    return env


# Simplified SB3 environment wrapper
def make_env_sb3(render_mode=None):
    # env = make_env(ENV_NAME, render_mode=render_mode)
    env = make_env(ENV_NAME, render_mode=render_mode)
    # env = VecFrameStack(env, n_stack=4)
    return Monitor(env, allow_early_resets=True)


# Model evaluation function
# def eval_model(env, model_name):
#     print(f"Loading and evaluating model '{model_name}'...")
#     model = None
#     if model_name == "dqn":
#         model = DQN.load(config["export_path"] + model_name)
#     elif model_name == "a2c":
#         model = A2C.load(config["export_path"] + model_name)
#     elif model_name == "ppo":
#         model = PPO.load(config["export_path"] + model_name)
#     elif model_name == "sac":
#         model = SAC.load(config["export_path"] + model_name)

#     mean_rewards = []
#     for _ in tqdm(range(100)):
#         obs, _ = env.reset()
#         total_reward = 0
#         while True:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, _ = env.step(action)
#             if terminated or truncated:
#                 break
#             total_reward += reward
#         mean_rewards.append(total_reward)

#     mean_reward = np.mean(mean_rewards)
#     std_reward = np.std(mean_rewards)
#     print(f"Reward (mean ± std): {mean_reward:.2f} ± {std_reward:.4f}")


# def custom_epsilon_schedule(progress_remaining: float) -> float:
#     return max(EPS_MIN, progress_remaining * (EPS_START - EPS_MIN) + EPS_MIN)



class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval = 0

    def _on_step(self) -> bool:
        if self.n_calls - self.last_eval >= self.eval_freq:
            self.last_eval = self.n_calls
            mean_reward, std_reward = self.evaluate_policy()
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")
        return True

    def evaluate_policy(self):
        rewards = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()  # VecEnv API: Only returns `obs`
            total_reward = 0.0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                total_reward += reward
                if np.all(done):  # VecEnv returns a list of `done`
                    break
            rewards.append(total_reward)
        return np.mean(rewards), np.std(rewards)



# Model training function
def train_model(env, model_name):
    run = wandb.init(
        project="Tennis",
        config=config,
        name=model_name,
        sync_tensorboard=True,
        save_code=True,
    )

    eval_env = make_env_sb3()  # Separate evaluation environment
    eval_callback = CustomEvalCallback(eval_env, eval_freq=10000, n_eval_episodes=5, verbose=1)

    print(f"\n>>> Creating and training model '{model_name}'...")
    model = None
    if model_name == "dqn":
        model = DQN(config["policy_type"], env, verbose=0, tensorboard_log=f"runs/{run.id}", 
                    learning_rate=1e-2, 
                    exploration_fraction=0.6)
    elif model_name == "a2c":
        model = A2C(config["policy_type"], env, verbose=0, tensorboard_log=f"runs/{run.id}")
    elif model_name == "ppo":
        model = PPO(config["policy_type"], env, verbose=0, tensorboard_log=f"runs/{run.id}")
    elif model_name == "sac":
        model = SAC(config["policy_type"], env, verbose=0, tensorboard_log=f"runs/{run.id}")

    video_callback = VideoRecorderCallback(video_folder, video_freq=1, video_length=1000)
    model.learn(total_timesteps=config["total_timesteps"], callback=[WandbCallback(verbose=2), eval_callback, video_callback])

    model.save(config["export_path"] + model_name)
    print(f"Model exported at '{config['export_path']}{model_name}'")
    run.finish()



if __name__ == "__main__":
    # Configuration
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1000000,
        "env_name": "CartPole-v1",
        "export_path": "./exports/",
    }

    ENV_NAME = "ALE/Tennis-v5"

    # Video recording setup
    video_folder = "./videos/"
    os.makedirs(video_folder, exist_ok=True)

    env = make_env_sb3(render_mode='rgb_array')

    env.metadata["render_fps"] = 15

    # Train and evaluate models
    models = ["a2c"]
    print("Starting training...")
    for model_name in models:
        train_model(env, model_name)
