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
import torch

from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from PIL import Image, ImageDraw, ImageFont

import os
import ale_py

from stable_baselines3.common.callbacks import BaseCallback
import yaml


gym.register_envs(ale_py)
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

from tqdm.notebook import tqdm
from stable_baselines3.common.callbacks import BaseCallback

class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None
    
    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])
    
    def _on_step(self):
        self.progress_bar.update(1)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None

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
    def __init__(self, learning_starts, eval_env, eval_freq=1, verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval = 0
        self.learning_starts = learning_starts

    def _on_step(self) -> bool:
        if self.n_calls - self.last_eval >= self.eval_freq:
            self.last_eval = self.n_calls

            start_time = datetime.strptime(time, "%d%m-%H%M%S")
            current_time = datetime.strptime(datetime.now().strftime("%d%m-%H%M%S"), "%d%m-%H%M%S")

            # Calculate the elapsed time
            elapsed_time = current_time - start_time
            hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)

            # Print the result
            print(f"Step {self.n_calls}: Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds", flush=True)
            if self.n_calls < self.learning_starts:
                print(f"Step {self.n_calls}: Skipping evaluation as learning has not started yet")
                return True
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
        
        save_gif = f"{export_path}/videos/eval-{calls}.gif"
        wandb.log({"eval_reward": total_reward})
        images[0].save(save_gif, save_all=True, append_images=images[1:], duration=50, loop=0) 
        return float(total_reward), 0.0

# Simplified SB3 environment wrapper
def make_env_sb3(render_mode=None, val=False):
    env = make_env(ENV_NAME)
    env = Monitor(env)
    return env

from stable_baselines3.common.vec_env import VecEnvWrapper

class TennisRewardVecWrapper(VecEnvWrapper):
    def __init__(self, venv, penalty=-0.1):
        super(TennisRewardVecWrapper, self).__init__(venv)
        self.penalty = penalty

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # Penalize for 'noop' actions (if applicable)
        modified_rewards = [
            reward + self.penalty if info.get("noop_action", False) else reward
            for reward, info in zip(rewards, infos)
        ]
        return obs, modified_rewards, dones, infos

def return_model(cfg, env):
    send_cfg = cfg['model_arguments'] if 'model_arguments' in cfg else {}
    if cfg['model_name'] == 'ppo':
        print("Creating PPO model")
        model = PPO(cfg['policy_type'], env, **send_cfg)
    elif cfg['model_name'] == 'dqn':
        print("Creating DQN model")
        model = DQN(cfg['policy_type'], env, **send_cfg)
    elif cfg['model_name'] == 'a2c':
        print("Creating A2C model")
        model = A2C(cfg['policy_type'], env, **send_cfg)
    elif cfg['model_name'] == 'sac':
        print("Creating SAC model")
        model = SAC(cfg['policy_type'], env, **send_cfg)
    else:
        raise ValueError("Model name not recognized")
    return model

import sys
ENV_NAME = "ALE/Tennis-v5"
if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python SB3_Tennis.py <config_file>"
    assert os.path.exists(sys.argv[1]), f"Config  {sys.argv[1]} not found"
    device = "cpu"
    print("Loading configuration from:", sys.argv[1])
    global time
    global export_path
    time = datetime.now().strftime("%d%m-%H%M%S")

    with open(sys.argv[1], "r") as file:
        config = yaml.safe_load(file)
    
    model_name = f"{config['model_name']}-{time}"
    config['export_path'] = f"/home/nbiescas/Project_RL/Tennis/runs/{model_name}"
    export_path = config['export_path']

    #Save the yaml
    os.makedirs(config['export_path'])
    os.makedirs(f"{config['export_path']}/videos/", exist_ok=True)
    with open(f"{config['export_path']}/config.yaml", "w") as file:
        yaml.dump(config, file)
    # Create environment
    env = make_atari_env("ALE/Tennis-v5", n_envs=config['n_envs'], seed=0)
    env = TennisRewardVecWrapper(env)
    # Train and evaluate models
    run = wandb.init(project="Tennis", 
                     group=config['model_name'],
                     config=config, 
                     name=model_name, 
                     sync_tensorboard=True, 
                     save_code=True, 
                     monitor_gym=True,)
    #config['tensorboard_log'] = f"runs/{run.id}"

    # Callbacks
    if 'model_arguments' in config:
        learning_starts = config['model_arguments'].get('learning_starts', 0)
    else:
        learning_starts = 0
    print(f"Learning starts at: {learning_starts}")
    eval_callback = CustomEvalCallback(eval_env=env, learning_starts = learning_starts, eval_freq=config['eval_freq'], verbose=2)
    tqdm_callback = TqdmCallback()
    callback = CallbackList([eval_callback, tqdm_callback, WandbCallback()])

    print(f'\n>>> Creating and training model {config["model_name"]}...')
    print("Saving logs at:", config['export_path'])

    model = return_model(config, env)
    print("Learning started")
    model.learn(total_timesteps=config["total_timesteps"], callback = callback)

    model.save(os.path.join(config["export_path"], model_name))
    print(f"Model exported at '{config['export_path']}/{model_name}'")
    run.finish()
