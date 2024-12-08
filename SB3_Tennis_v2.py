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
import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, ReshapeObservation, FrameStackObservation
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from gymnasium import RewardWrapper

from stable_baselines3.common.callbacks import EvalCallback

from PIL import Image, ImageDraw, ImageFont

import os
import ale_py

from stable_baselines3.common.callbacks import BaseCallback

# Custom Observation Wrappers
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name, skip=4, stack_size=2, reshape_size=(84, 84), render_mode="rgb_array"):
    env = gym.make(env_name, render_mode=render_mode)  # Explicitly set render_mode to "rgb_array"
    env = MaxAndSkipObservation(env, skip=skip)
    env = ResizeObservation(env, reshape_size)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ImageToPyTorch(env)
    env = ReshapeObservation(env, reshape_size)
    env = FrameStackObservation(env, stack_size=stack_size)
    env = ScaledFloatFrame(env)
    # env = ActionPenaltyWrapper(env, max_consecutive_zeros=2, penalty_reward=-50.0)
    return env


def make_env_eval(env_name, skip=4, stack_size=4, reshape_size=(84, 84), render_mode="rgb_array"):
    env = gym.make(env_name, render_mode=render_mode)  # Explicitly set render_mode to "rgb_array"
    env = ImageToPyTorch(env)
    env = ReshapeObservation(env, reshape_size)
    env = FrameStackObservation(env, stack_size=stack_size)
    env = ScaledFloatFrame(env)
    # env = ActionPenaltyWrapper(env, max_consecutive_zeros=2, penalty_reward=-50.0)
    return env

class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval = 0

    def _on_step(self) -> bool:
        if self.n_calls - self.last_eval >= self.eval_freq:
            self.last_eval = self.n_calls
            mean_reward, std_reward = self.evaluate_policy(self.n_calls)
            print(f"Step {self.n_calls}: Evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")
        return True

    def evaluate_policy(self, calls):
        global time
        obs = self.eval_env.reset()
        images = []
        visualize = True
        total_reward = 0.0
        for t in range(1000):
            img = env.render()
                
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _ = self.eval_env.step(action)
            total_reward += reward
            
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil) 
            
            # Add the step number to the upper-left corner of the image
            font = ImageFont.load_default() 
            draw.text((10, 10), f"Step: {t}", fill="white", font=font)
            
            # Add the epoch number centered at the bottom of the image
            epoch_text = f"Epoch: {calls}"
            text_bbox = draw.textbbox((0, 0), epoch_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((img_pil.width - text_width) // 2, img_pil.height - text_height - 10), epoch_text, fill="white", font=font)

            # Take a step in the environment using the selected action
            state, reward, is_done, _ = env.step(action)
            
            # Add the current reward to the top-right corner of the image
            total_reward += float(reward)
            reward_text = f"Reward: {float(total_reward):.2f}"
            text_bbox = draw.textbbox((0, 0), reward_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((img_pil.width - text_width - 10, 10), reward_text, fill="white", font=font)
            
            images.append(img_pil)
            # is_done = terminated or truncated    
            if is_done:
                break
        
        save_gif = f"videos-{time}/eval-{calls}.gif"
        os.makedirs(f"videos-{time}", exist_ok=True)
        images[0].save(save_gif, save_all=True, append_images=images[1:], duration=50, loop=0) 
        return float(total_reward), 0.0




# Simplified SB3 environment wrapper
def make_env_sb3(render_mode=None, val=False):
    env = make_env(ENV_NAME)
    env = Monitor(env)
    return env


# Model training function
# def train_model(env, model_name, eval_callback):
    

class ActionPenaltyWrapper(RewardWrapper):
    def __init__(self, env, max_consecutive_zeros=5, penalty_reward=-10.0):
        super().__init__(env)
        self.max_consecutive_zeros = max_consecutive_zeros
        self.penalty_reward = penalty_reward
        self.consecutive_zeros = 0

    def reset(self, **kwargs):
        self.consecutive_zeros = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        if action == 0:
            self.consecutive_zeros += 1
        else:
            self.consecutive_zeros = 0
        
        if self.consecutive_zeros > self.max_consecutive_zeros:
            reward += self.penalty_reward  # Apply penalty
        
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # Configuration
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1000000,
        "export_path": "./exports/",
    }

    ENV_NAME = "ALE/Tennis-v5"

    global time
    time = datetime.now().strftime("%d%m-%H%M%S")

    # Video recording setup
    video_folder = "./videos/"
    os.makedirs(video_folder, exist_ok=True)

    # print(make_env_sb3(render_mode='rgb_array'))

    env = DummyVecEnv([lambda: make_env_sb3(render_mode="rgb_array")])  # Ensure render_mode is passed as 'rgb_array'

    eval_env = DummyVecEnv([lambda: make_env_sb3(render_mode="rgb_array")])  # Separate evaluation environment
    # eval_callback = CustomEvalCallback(eval_env, eval_freq=1000, verbose=2)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=False, render=False)

    # env.metadata["render_fps"] = 15

    # Train and evaluate models
    model_name = "dqn"
    print("Starting training...")
    run = wandb.init(project="Tennis", config=config, 
                     name=model_name, sync_tensorboard=True, 
                     save_code=True, monitor_gym=True,)

    print(f"\n>>> Creating and training model '{model_name}'...")
    if model_name == "ppo":
        model = PPO(config["policy_type"], env, verbose=0, gae_lambda=0.1, ent_coef=1, tensorboard_log=f"runs/{run.id}")
    
    elif model_name == "dqn":
        model = DQN(config["policy_type"], env, 
                    verbose=0, exploration_fraction=0.9, 
                    tensorboard_log=f"runs/{run.id}")
    
    model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback())

    model.save(config["export_path"] + model_name)
    print(f"Model exported at '{config['export_path']}{model_name}'")
    run.finish()