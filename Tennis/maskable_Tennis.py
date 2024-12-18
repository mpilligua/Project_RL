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

from PIL import Image, ImageDraw, ImageFont
import collections

import os

import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
import yaml


import seaborn
from wrappers import *
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, ReshapeObservation
from stable_baselines3.common.atari_wrappers import *

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib import QRDQN



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
        return obs[:, self.top:self.top+self.height, self.left:self.left+self.width]

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

# Simplified SB3 environment wrapper
def make_env_sb3(config, evaluation=False):
    env = make_env(**config)
    env.reset()

    if not evaluation:
        env = Monitor(env)
    return env

def make_env(env_name, skip=4, stack_size=4, reshape_size=(84, 84), render_mode=None, eval=False):
    env = gym.make(env_name, render_mode=render_mode)
    log("Standard Env.        : {}".format(env.observation_space.shape))
    if skip > 1:
        env = MaxAndSkipObservation(env, skip=skip)
        log("MaxAndSkipObservation: {}".format(env.observation_space.shape))

    env = ResizeObservation(env, reshape_size)
    log("ResizeObservation    : {}".format(env.observation_space.shape))
    env = ImageToPyTorch(env)
    log("ImageToPyTorch       : {}".format(env.observation_space.shape))
    env = Keep_red_dim(env)
    log("Keep_red_dim         : {}".format(env.observation_space.shape))
    env = ReshapeObservation(env, reshape_size)
    log("ReshapeObservation   : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=stack_size)
    log("FrameStackObservation: {}".format(env.observation_space.shape))
    #env = CropObservation(env, 5, 6, 80, 80)    
    #log("CropObservation      : {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    log("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    env = ForceDifferentAction(env)
    log("ForceDifferentAction : {}".format(env.observation_space.shape))
    #env = RewardLongPoints(env, return_added_reward=eval)
    #log("RewardLongPoints     : {}".format(env.observation_space.shape))
    
    if not eval:
        env = EndWhenLosePoint(env, return_added_reward=eval)
        log("EndWhenLosePoint     : {}".format(env.observation_space.shape))
    
    env.spec.reward_threshold = 100000000
    return env


def make_env_eval(env_name, skip=4, stack_size=4, reshape_size=(84, 84), eval = False, render_mode="rgb_array"):
    env = gym.make(env_name, render_mode=render_mode)  # Explicitly set render_mode to "rgb_array"
    env = ImageToPyTorch(env)
    env = ReshapeObservation(env, reshape_size)
    env = FrameStackObservation(env, stack_size=stack_size)
    env = ScaledFloatFrame(env)
    # env = ActionPenaltyWrapper(env, max_consecutive_zeros=2, penalty_reward=-50.0)
    return env



class CustomEvalCallback(WandbCallback):
    def __init__(self, eval_env, export_path=None, eval_freq=1, verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval = 0
        self.export_path = export_path

    def _on_step(self) -> bool:
        # Call parent _on_step to log WandB metrics
        super()._on_step()

        # Perform custom evaluation
        if self.n_calls - self.last_eval >= self.eval_freq:
            self.last_eval = self.n_calls
            mean_reward, std_reward = self.evaluate_policy(self.n_calls)
            print(f"Step {self.n_calls}: Evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")
            wandb.log({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward}, step=self.n_calls)

        return True
    
    def evaluate_policy(self, calls, policy='train'):
        current_state = self.eval_env.reset()[0]
    
        done = False
        images = []
        total_reward = 0.0
        t = 0
        keep = 0
        const_text_height = 12
        colors = seaborn.color_palette("coolwarm", as_cmap=True)

        while not done and t < 1000:
            img = self.eval_env.render()
            img = img[4:, 8:]
            img_pil = Image.fromarray(img)

            #action, q_values = self.get_action(current_state, mode='train', epsilon=epsilon, return_q_values=True)
            action_masks = self.eval_env.action_mask()
            action = self.model.predict(current_state, action_masks=action_masks)[0]
            #current_state, reward, added_reward, terminated, truncated,  = self.eval_env.step(action)
            #img_array = current_state[:, :, 0]
            img_array = current_state[0, :, :]
            
            #img_array = np.expand_dims(current_state[0], 0)
            #img_array = np.repeat(img_array, 3, axis=0)
            #img_array = np.moveaxis(img_array, 0, -1)  # Move the channel axis to the end
            #img_array = (img_array * 255).astype(np.uint8)  # Convert to uint8
            img_input_model = Image.fromarray(img_array)

            # Ensure both images have the same height
            height = max(img_pil.height, img_input_model.height)
            img_pil = img_pil.resize((img_pil.width, height))
            new_width = int(img_input_model.width * (height / img_input_model.height))
            img_input_model = img_input_model.resize((new_width, height))

            # Calculate the dimensions for the final image
            imgs_width = img_pil.width + (img_input_model.width * 4)
            combined_height = height + const_text_height
            #q_values_width = (combined_height-const_text_height) // 18	

            total_width = imgs_width# + q_values_width

            # Create a blank canvas with extra space for text
            combined_image = Image.new("RGB", (total_width, combined_height), color=(0, 0, 0))

            # Paste the images onto the canvas
            combined_image.paste(img_pil, (0, const_text_height))

            for i in range(3, -1, -1):
                img_array = current_state[i, :, :]
                img_input_model = Image.fromarray(img_array)
                img_input_model = img_input_model.resize((new_width, height))
                combined_image.paste(img_input_model, (img_pil.width + (i * img_input_model.width), const_text_height))
            
            #combined_image.paste(img_input_model, (img_pil.width, const_text_height))

            # Add a black bar with the text above the images
            box = Image.new("RGB", (total_width, const_text_height), color=(0, 0, 0))
            combined_image.paste(box, (0, 0))

            draw = ImageDraw.Draw(combined_image)

            # Add the step number to the upper-left corner of the image
            font = ImageFont.load_default() 
            draw.text((10, 1), f"Step: {t}", fill="white", font=font)

            # Add the epoch number centered at the top of the image
            #epoch_text = f"Episode: {self.episode}"
            #text_bbox = draw.textbbox((0, 0), epoch_text, font=font)
            #text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            #draw.text(((combined_image.width - text_width) // 2, 1), epoch_text, fill="white", font=font)
    

            current_state, reward, terminated, truncated, info = self.eval_env.step(action)

            # Add the current reward to the top-right corner of the image
            total_reward += reward
            reward_text = f"Reward: {total_reward:.2f}"
            text_bbox = draw.textbbox((0, 0), reward_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((combined_image.width - text_width - 10, 1), reward_text, fill="white", font=font)
            
            # Add the added reward in the top-right corner of the image below the reward
            #if added_reward != 0 or keep:
            #    if keep == 0:
            #        keep = 6
            #        added_reward_text = f"+{added_reward:.2f}"
            #        prev_added_reward = added_reward
            #    else:
            #        added_reward_text = f"+{prev_added_reward:.2f}"
            #    
            #    text_bbox = draw.textbbox((0, 0), added_reward_text, font=font)
            #    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            #    draw.text((combined_image.width - text_width - 10, 17), added_reward_text, fill="white", font=font)
            #    keep -= 1

            # Add the action taken to the top-left corner of the image below the steps
            current_score = self.eval_env.current_score
            current_score_text = f"Score: {current_score[0]} - {current_score[1]}"
            text_bbox = draw.textbbox((0, 0), current_score_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width + 40) // 3, 1), current_score_text, fill="white", font=font)

            current_serve = self.eval_env.server
            current_serve_text = f"Server: {current_serve}"
            text_bbox = draw.textbbox((0, 0), current_serve_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width + 40) // 2, 1), current_serve_text, fill="white", font=font)

            action_text = f"Act: {action}"
            text_bbox = draw.textbbox((0, 0), action_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width) // 4, 1), action_text, fill="white", font=font)

            done = terminated or truncated    
            total_reward += reward  

            images.append(combined_image)

            t += 1

        # save the image
        wandb.log({"eval_reward": total_reward //2}, step=calls)
        self.model.save(f"{self.export_path}/models/{self.n_calls}_")
        # Keep the last 5 save models only
        sorted_models = sorted(os.listdir(f"{self.export_path}/models/"), key=lambda x: int(x.split("_")[0]))
        if len(sorted_models) > 5:
            os.remove(f"{self.export_path}/models/{sorted_models[0]}")


        images[0].save(f"{self.export_path}/videos/{self.n_calls}_.gif", save_all=True, append_images=images[1:], duration=100, loop=0)

        self.eval_env.close()
        return total_reward // 2, 0

    def prev_evaluate_policy(self, calls):
        global time
        obs = self.eval_env.reset()[0]
        images = []
        total_reward = 0.0
        
        for i in range(1000):
 
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            images.append(Image.fromarray(self.eval_env.render()))
            if done:
                break
        
        save_gif = f"{self.export_path}/videos/eval-{calls}.gif"
        wandb.log({"eval_reward": total_reward}, step=calls)
        images[0].save(save_gif, save_all=True, append_images=images[1:], duration=100, loop=0) 
        return float(total_reward), 0.0

    
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, device="cuda"):
        super().__init__(observation_space, features_dim)

        self.device = device

        print("The observation space is", observation_space.shape)

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        self.to(device)

    def forward(self, x):
        x = self.network(x.to(self.device))

        #if x.dim() > 2:
        #    x = x.view(x.size(0), -1)
        #else:
        #    x = x.view(1, -1)
        return x

class DQN_NET(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, device="cuda", dueling_layer=False):
        super().__init__(observation_space, features_dim)

        self.device = device

        print("The observation space is", observation_space.shape)
        input_channels = observation_space.shape[0] if len(observation_space.shape) == 3 else 1
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, features_dim)
        self.relu = nn.ReLU()

        self.to(device)

    def forward(self, x):
        x = self.conv1(x.to(self.device))
        x = self.conv2(self.relu(x))
        x = self.conv3(self.relu(x))
        x = self.conv4(self.relu(x))
        x = self.avgpool(self.relu(x))

        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        else:
            x = x.view(1, -1)
        
        x = self.linear1(self.relu(x))
        x = self.linear2(self.relu(x))

        return x

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
        print(observations)
        x = self.cnn(observations)

        return self.cnn(observations)
    
def return_model(cfg, env, policy_kwargs=None):
    if cfg['model_name'] == 'ppo':
        log("Using PPO model")
        model = PPO(policy = cfg['policy_type'],
                    env = env,
                    learning_rate = cfg['learning_rate'],
                
                    policy_kwargs=policy_kwargs,  # POlicy kwargs
                    verbose=1)
    elif cfg['model_name'] == 'MaskablePPO':
        model = MaskablePPO(policy = cfg['policy_type'], 
                    env = env, 
                    learning_rate=cfg['learning_rate'], 
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=f"runs/{name_run}")
        
    elif cfg['model_name'] == 'dqn':
        model = DQN(cfg['policy_type'], env, **cfg['model_arguments'])
    
    elif cfg['model_name'] == 'a2c':
        model = A2C(cfg['policy_type'], env)
    elif cfg['model_name'] == 'sac':
        model = SAC(cfg['policy_type'], env)
    else:
        raise ValueError("Model name not recognized")
    return model

def log(msg):
    logger.info(msg)
    print(msg, flush=True)



def make_env(num_envs=1, skip=4, stack_size=4, reshape_size=(84, 84), eval=False):
    env = gym.make("ALE/Tennis-v5", render_mode="rgb_array")
    # log("Standard Env.        : {}".format(env.observation_space.shape))
    # if skip > 1:
    #     env = MaxAndSkipObservation(env, skip=skip)
    #     log("MaxAndSkipObservation: {}".format(env.observation_space.shape))
    env = ResizeObservation(env, reshape_size)
    # #log("ResizeObservation    : {}".format(env.observation_space.shape))
    #env = GrayscaleObservation(env, keep_dim=True)
    # # log("GrayscaleObservation : {}".format(env.observation_space.shape))
    env = ImageToPyTorch(env)
    # log("ImageToPyTorch       : {}".format(env.observation_space.shape))
    env = ReshapeObservation(env, reshape_size)
    # log("ReshapeObservation   : {}".format(env.observation_space.shape))
    env = Keep_red_dim(env)
    #log("Keep_red_dim         : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=stack_size)
    #log("FrameStackObservation: {}".format(env.observation_space.shape))
    # # env = CropObservation(env, 5, 6, 80, 80)    
    # # log("CropObservation      : {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    # log("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    #env = ForceDifferentAction(env)
    # # log("ForceDifferentAction : {}".format(env.observation_space.shape))
    env = RewardLongPoints(env, return_added_reward=eval)
    #log("RewardLongPoints     : {}".format(env.observation_space.shape))
    if not eval:
        env = EndWhenLosePoint(env)
    # log("EndWhenLosePoint     : {}".format(env.observation_space.shape))
    env = MapActions(env)
    env = GetBallPosition(env)
    # env = ForceFireOnFirstMove(env)

    env = Monitor(env)
    # env.spec.reward_threshold = 100000000
    return env

def make_env_actions(env_id, seed, idx, capture_video, run_name, render=False):
    def thunk():
        render_mode = "rgb_array" if render else "rgb_array"
        env = gym.make(env_id, render_mode=render_mode)

        #env = gym.wrappers.RecordEpisodeStatistics(env)
        #if capture_video:
        #    if idx == 0:
        #        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = ImageToPyTorch(env)
        env = ReshapeObservation(env, (3, 84, 84))
        #env = Keep_red_dim(env)
        env = MaxAndSkipEnv(env, skip=4) #Aixo fa els skip de 4 i pille els dos ultimes i fa merge amb np.max aixo es lo que se considera frame
        env = FrameStackObservation(env, 4) #Aixo fa stack de 4 frames
        env = CropObservation(env, 5, 6, 80, 80)  
        env = ScaledFloatFrame(env)
        #env = RewardLongPoints(env, return_added_reward=eval)
        #log("RewardLongPoints     : {}".format(env.observation_space.shape))
        if not eval:
            env = EndWhenLosePoint(env)
        # log("EndWhenLosePoint     : {}".format(env.observation_space.shape))
        env = MapActions(env)
        #env = ForceFireOnFirstMove(env)
        #env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = ClipRewardEnv(env)
        env = TennisWrapper2(env)
        env = Monitor(env)


        env = GetBallPosition(env)

        return env
    
    # env.spec.reward_threshold = 100000000
    return thunk


class TennisWrapper2(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)[0]
        self.current_score = [0,0]
        self.score = [0,0] 
        self.all_score = [0,0] 
        self.server = 0 
        #self.run_length = 0

        obs = self.run_reset(obs)
        return obs, {}

    def action_mask(self):
        am = np.array([True]*self.action_space.n)
        if self.server == 0 and self.run_length < 30:
            am[[0] + list(range(2,10))] = False
        return am

    def run_reset(self,old_obs,max_loop=1000):
        self.run_length = 0
        for _ in range(max_loop):
            obs = self.env.step(0)[0]
            if not np.all(old_obs == obs):
                break
            old_obs = obs
        return obs
        
    def step(self, action):
        assert self.action_mask()[action], "not a legal action"

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.run_length += 1
        info["run_done"] = False        

        if reward != 0:
            info["run_done"] = True
            info["run_length"] = self.run_length
            #reward += info["run_length"] / 50
            run_winner = 0 if reward == 1 else 1
            self.current_score[run_winner] += 1
            self.all_score[run_winner] += 1
       
            if self.current_score[run_winner] >= 4 and self.current_score[run_winner] - self.current_score[1-run_winner]>=2:      
                self.current_score = [0,0]
                self.score[run_winner] += 1
                self.server = 1 - self.server 
            obs = self.run_reset(obs.copy())
            #self.run_length = 0
        info["action_mask"] = self.action_mask()


        return obs, reward, terminated, truncated, info


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.

    return env.action_mask()
    
if __name__ == "__main__":
    import os
    import time
    from configs.SB3 import list_of_wrappers, sac_arguments, ac2_arguments, training_config, ppo_arguments, train_env_config, eval_env_config# <--- All hyperparameters are defined here
    import logging
    
    idx = time.strftime("%d%H%M%S")
    
    name_run = f"TennisAllPPO"

    print("Overwriting the run name")
    
    run = wandb.init(project="Freeway", name=name_run, entity="pilligua2", group="Tennis", sync_tensorboard=True)

    results_dir = f"/ghome/mpilligua/RL/Project_RL/Tennis/runs/{name_run}"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/videos", exist_ok=True)
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    
    # initialize logging file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename=f"{results_dir}/log.log",
    )
    logger = logging.getLogger(__name__)
    
    log(msg=f"Saving to: {results_dir}")
    log(f"Wandb run: {name_run} - {run.id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ENV_NAME = "Tennis-v4"
    
    def action_masker():
        env = make_env_actions(env_id=ENV_NAME, seed=0, idx=0, capture_video=False, run_name=name_run, render=False)()
        ev = ActionMasker(env, mask_fn)
        return ev
    
    train_env = make_vec_env(action_masker, n_envs=8)
    eval_env = make_env_actions(env_id=ENV_NAME, seed=0, idx=0, capture_video=False, run_name=name_run, render=True)()

    eval_callback = CustomEvalCallback(eval_env, export_path = results_dir, eval_freq=10000, verbose=2)
    # callback = CallbackList([eval_callback, ()])
    
    policy_kwargs = dict()
    if training_config['custom_dqn']:
        policy_kwargs['features_extractor_class'] = CustomCNN
        policy_kwargs['features_extractor_kwargs'] = dict(features_dim=32, device=device)
        
    training_config['model_arguments'] = ppo_arguments


    wandb.config.update(training_config)


    #model = QRDQN("CnnPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1)

    model = MaskablePPO("MultiInputPolicy", 
                    env = train_env, 
                    learning_rate=training_config['model_arguments']['learning_rate'], 
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=f"{results_dir}",
                    n_steps=128,
                    n_epochs=4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.1,
                    clip_range_vf=None,

                    normalize_advantage=True,  # Normalize advantages for stability
    ent_coef=0.01,  # Encourage exploration with entropy regularization
    vf_coef=0.5,  # Standard value function coefficient
    max_grad_norm= 0.5,  # Clip gradients to avoid large updates
#
    target_kl = 0.01)  # Target KL divergence to avoid overly aggressive updates

    model.learn(training_config["total_timesteps"], callback=eval_callback)
    model.save(f"{results_dir}/maskable_tennis")



