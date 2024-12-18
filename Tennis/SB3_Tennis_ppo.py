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

from stable_baselines3.common.callbacks import EvalCallback, CallbackList

import os
import time
from configs.SB3 import list_of_wrappers, sac_arguments, ac2_arguments, training_config, ppo_arguments, train_env_config, eval_env_config# <--- All hyperparameters are defined here
import logging


from PIL import Image, ImageDraw, ImageFont
import collections

import os
import ale_py
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
import yaml


import seaborn
from wrappers import *

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import FireResetEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, ReshapeObservation
from stable_baselines3.common.atari_wrappers import *
from wrappers import *

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def visualize_train(env, ppo_model, episode =100000, save_path = ""):
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        current_state, _ = env.reset()
    
        done = False
        images = []
        total_reward = 0.0
        t = 0
        const_text_height = 12

        while not done and t < 1000:
            img = env.render()
            # img = img/255
            img = img.astype(np.uint8)
            # print(img.shape, img.max(), img.min(), img.dtype)
            img_pil = Image.fromarray(img)

            # Make a step in the environment
            action, _ = ppo_model.predict(current_state)
            # print(action)
            current_state, reward, truncated, termianted, _ = env.step(action)            
            
            img_array = current_state[0]
            img_array = np.expand_dims(current_state[0], 0)
            img_array = np.repeat(img_array, 3, axis=0)
            img_array = np.moveaxis(img_array, 0, -1)  # Move the channel axis to the end
            img_array = (img_array * 255).astype(np.uint8)  # Convert to uint8
            img_array = img_array[:, :, -1]
            # print(img_array.shape, img_array.max(), img_array.min(), img_array.dtype)
            img_input_model = Image.fromarray(img_array)

            # Ensure both images have the same height
            height = max(img_pil.height, img_input_model.height)
            img_pil = img_pil.resize((img_pil.width, height))
            new_width = int(img_input_model.width * (height / img_input_model.height))
            img_input_model = img_input_model.resize((new_width, height))

            # Calculate the dimensions for the final image
            imgs_width = img_pil.width + img_input_model.width
            combined_height = height + const_text_height
            q_values_width = (combined_height-const_text_height) // 18	

            total_width = imgs_width + q_values_width

            # Create a blank canvas with extra space for text
            combined_image = Image.new("RGB", (total_width, combined_height), color=(0, 0, 0))

            # Paste the images onto the canvas
            combined_image.paste(img_pil, (0, const_text_height))
            combined_image.paste(img_input_model, (img_pil.width, const_text_height))

            # Add a black bar with the text above the images
            box = Image.new("RGB", (total_width, const_text_height), color=(0, 0, 0))
            combined_image.paste(box, (0, 0))

            draw = ImageDraw.Draw(combined_image)

            # Add the step number to the upper-left corner of the image
            font = ImageFont.load_default() 
            draw.text((10, 1), f"Step: {t}", fill="white", font=font)
            
            # Add the epoch number centered at the top of the image
            epoch_text = f"Episode: {episode}"
            text_bbox = draw.textbbox((0, 0), epoch_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width) // 2, 1), epoch_text, fill="white", font=font)
        
            # Add the current reward to the top-right corner of the image
            # print(reward[0])
            total_reward += reward
            # print(total_reward)
            reward_text = f"Reward: {total_reward:.2f}"
            text_bbox = draw.textbbox((0, 0), reward_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((combined_image.width - text_width - 10, 1), reward_text, fill="white", font=font)

            # Add the action taken to the top-left corner of the image below the steps
            action_text = f"Act: {action}"
            text_bbox = draw.textbbox((0, 0), action_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width) // 4, 1), action_text, fill="white", font=font)

            done = truncated    
            # total_reward += reward  

            images.append(combined_image)

            t += 1

        # save the image
        images[0].save(save_path, save_all=True, append_images=images[1:], duration=120, loop=0)

        env.close()
        print(f"Total reward: {total_reward}")
        return total_reward
    

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We have H, W, C
        n_input_channels = 4
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations = observations.permute(0, 2, 3, 1)
        # print(observations.shape)
        return self.cnn(observations)



# Define a custom callback for evaluation and saving the model
class EvalCallback(WandbCallback):
    def __init__(self, eval_env, eval_freq, save_path, log_freq=0, verbose=0):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.log_freq = log_freq
        self.save_freq = 25000
        self.super = super()

    def _init_callback(self):
        self.super._init_callback()
        # Create the directory to save models if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _on_step(self) -> bool:
        print("Step", self.n_calls)
        self.super._on_step()
        # Evaluate the model every `eval_freq` steps
        if self.n_calls % self.eval_freq == 0:
            print(f"Step {self.n_calls}: Evaluating the model...")
            save_path = f"{self.save_path}/videos/ppo_pong_{self.n_calls}.gif"
            visualize_train(self.eval_env, self.model, episode=self.n_calls, save_path=save_path)
            wandb.log({"video":wandb.Image(save_path, caption=f"Episode {self.n_calls}")})
            
        if self.n_calls % self.save_freq == 0:
            model.save(f"{self.save_path}/ppo_tenis.zip")
        
        return True



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

class ForceDifferentAction(gym.RewardWrapper):
    """
    If the last 10 actions are the same, give a really negative reward
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_actions = collections.deque(maxlen=15)

    def step(self, action):
        self.last_actions.append(action)
        state, reward, terminated, truncated, info = self.env.step(action)
        # print(f"Last actions: {self.last_actions}")
        if len(self.last_actions) == 15 and all(a == self.last_actions[0] for a in self.last_actions):
            return state, reward-100, True, truncated, info
        else:
            return state, reward, terminated, truncated, info

class RewardLongPoints(gym.RewardWrapper):
    """
    If the agent loses a point, give a positive reward based on the number of frames it took to lose the point
    """
    def __init__(self, env, return_added_reward=False):
        super().__init__(env)
        self.frames = 0
        self.last_reward = 0
        self.return_added_reward = return_added_reward

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        added_reward = 0
        if reward < self.last_reward:
            added_reward = min(self.frames, 150) / (150*2)
            log(f"Added reward: {added_reward}")
            reward += added_reward
            self.frames = 0
        else:
            self.frames += 1

        self.last_reward = reward
        if self.return_added_reward:
            return state, reward, added_reward, terminated, truncated, info
        else:
            return state, reward, terminated, truncated, info

class Keep_red_dim(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[1], old_shape[2]), dtype=np.float32)
                                                
    def observation(self, observation):
        # print(observation.shape)
        return observation[0]

class CropObservation(gym.ObservationWrapper):
    def __init__(self, env, top, left, height, width):
        super().__init__(env)
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(env.observation_space.shape[0], env.observation_space.shape[1], height, width), dtype=np.uint8)

    def observation(self, obs):
        return obs[:, :, self.top:self.top+self.height, self.left:self.left+self.width]

class ForceFireOnFirstMove(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frames = 0
    
    def step(self, action):
        if self.frames == 0:
            return self.env.step(1)
        else: 
            return self.env.step(action)



class EndWhenLosePoint(gym.RewardWrapper):
    """
    If the agent loses a point, give a positive reward based on the number of frames it took to lose the point
    """
    def __init__(self, env):
        super().__init__(env)
        self.frames = 0
        self.last_reward = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.frames += 1
        
        if reward > self.last_reward:
            reward += 100
            self.last_reward = reward
            print(f"Reward: {reward}")
            self.frames = 0
            return state, reward, True, truncated, info
        
        if reward < 0:
            # if self.frames < 50:
            #     reward = -10
            if self.frames > 150:
                reward = -50
            # else:
                # reward = min([self.frames/30, 5]) 
            # print(f"Reward: {reward}")
            self.frames = 0
            return state, reward, True, truncated, info

        self.last_reward = reward
        # print(f"Reward: {reward}")
        return state, reward, terminated, truncated, info

class MapActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(6)

    def action(self, action):
        return action

def make_env(num_envs=1, skip=4, stack_size=4, reshape_size=(84, 84), eval=False):
    env = gym.make("ALE/Tennis-v5", render_mode="rgb_array")
    log("Standard Env.        : {}".format(env.observation_space.shape))
    if skip > 1:
        env = MaxAndSkipObservation(env, skip=skip)
        log("MaxAndSkipObservation: {}".format(env.observation_space.shape))
        
    env = ResizeObservation(env, reshape_size)
    log("ResizeObservation    : {}".format(env.observation_space.shape))
    # env = GrayscaleObservation(env, keep_dim=True)
    # log("GrayscaleObservation : {}".format(env.observation_space.shape))
    env = ImageToPyTorch(env)
    log("ImageToPyTorch       : {}".format(env.observation_space.shape))
    env = ReshapeObservation(env, (3, 84, 84))
    log("ReshapeObservation   : {}".format(env.observation_space.shape))
    env = GrayscaleObservation(env)
    log("GrayscaleObservation : {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    log("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=stack_size)
    log("FrameStackObservation: {}".format(env.observation_space.shape))

    env.spec.reward_threshold = 100000000
    return env
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # train_env = make_env(num_envs=8, skip=4, stack_size=4, reshape_size=(84, 84), eval=False)
    eval_env = make_env(num_envs=1, skip=1, stack_size=4, reshape_size=(84, 84), eval=True)

    train_env = DummyVecEnv([lambda: make_env(num_envs=1, skip=4, stack_size=4, reshape_size=(84, 84), eval=False) for _ in range(12)])


    # train_env = VecFrameStack(train_env, n_stack=4)
    # eval_env = VecFrameStack(eval_env, n_stack=4)
    
    policy_kwargs = {
        'net_arch': dict(pi=[], vf=[]),
        'features_extractor_class': CustomCNN,
        'features_extractor_kwargs': {'features_dim': 32},
        'normalize_images': False
    }

    # Instantiate the agent on the specified device
    name_run = "TennisPPO3" # or "Tennis3"
    save_path = "/ghome/mpilligua/RL/Project_RL/Tennis/runs/" + name_run
    model = PPO("CnnPolicy", train_env, 
                device=device, 
                verbose=1, 
                policy_kwargs=policy_kwargs, 
                tensorboard_log=f"{save_path}/tensorboard/")
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/videos", exist_ok=True)
    print(f"Saving models to {save_path}")
    run = wandb.init(project="Freeway", entity="pilligua2", name=name_run, sync_tensorboard=True, group="Tennis")


    # print("PPO model", model.policy)
    # exit(0)
    batch_size = 2049
    
    eval_callback = EvalCallback(
        eval_env=eval_env,  # Use the same environment for evaluation
        eval_freq=batch_size,  # Evaluate every 50,000 steps
        save_path=save_path  # Path to save models
    )

    # Train the agent with a single call to `learn`
    model.learn(total_timesteps=10000000, callback=eval_callback)

    model.save(f"{save_path}/ppo_tenis.zip")
    wandb.finish()