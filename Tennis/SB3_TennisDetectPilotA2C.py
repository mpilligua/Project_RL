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
import ale_py
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
import yaml

import os
import time
from configs.SB3 import list_of_wrappers, sac_arguments, ac2_arguments, training_config, ppo_arguments, train_env_config, eval_env_config# <--- All hyperparameters are defined here
import logging


from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import FireResetEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, ReshapeObservation
from stable_baselines3.common.atari_wrappers import *
from wrappers import *
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
    
import seaborn
from wrappers import *
    

class CustomEvalCallback(WandbCallback):
    def __init__(self, eval_env, export_path=None, eval_freq=1, verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval = 0
        self.export_path = export_path
        self.super = super()
        
    def _on_step(self) -> bool:
        self.super._on_step()
        if self.n_calls - self.last_eval >= self.eval_freq:
            self.last_eval = self.n_calls
            mean_reward, std_reward = self.evaluate_policy(self.n_calls)
            print(f"Step {self.n_calls}: Evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")
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

        while not done and t < 10000:

            img = self.eval_env.render()
            img = img.astype(np.uint8)
            # img = img[4:, 8:]
            img_pil = Image.fromarray(img)

            #action, q_values = self.get_action(current_state, mode='train', epsilon=epsilon, return_q_values=True)
            
            #action_masks = self.eval_env.action_masks()
            action = self.model.predict(current_state)[0]#, action_masks=action_masks)[0]
            
            #current_state, reward, added_reward, terminated, truncated,  = self.eval_env.step(action)
            current_state, ball_position = current_state['masked_observation'], current_state['ball_position']
            ball_position = ball_position[0]  
            current_state = current_state[:1]
            img_array = current_state * 255.0
             
            #print(current_state.shape)
            #img_array = current_state[:, :, 0]
            #print(img_array.shape)
            #exit(0)
            #img_array = np.expand_dims(current_state[0], 0)
            img_array = np.repeat(img_array, 3, axis=0)
            img_array = np.moveaxis(img_array, 0, -1)  # Move the channel axis to the end
            
            img_array = img_array.astype(np.uint8)  # Convert to uint8
            img_input_model = Image.fromarray(img_array)

            # Put a red pixel where the ball is
            ball_position = ball_position[0]
            img_input_model.putpixel((ball_position[1], ball_position[0]), (255, 0, 0))

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
            # Draw a red point on the image where the ball is
            
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
            action_text = f"Act: {action}"
            text_bbox = draw.textbbox((0, 0), action_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width) // 4, 1), action_text, fill="white", font=font)

            done = terminated or truncated    
            total_reward += reward  

            images.append(combined_image)

            t += 1

        # save the image
        images[0].save(f"{self.export_path}/videos/{self.n_calls}_.gif", save_all=True, append_images=images[1:], duration=120, loop=0)
        wandb.log({'total_reward_eval': total_reward, "timestemps_eval": t})
        print(f"Total reward: {total_reward}")
        self.eval_env.close()
        return total_reward, 0


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We have H, W, C
        n_input_channels = 4
        self.dropout = nn.Dropout(0.5)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations = observations.permute(0, 2, 3, 1)
        # print(observations.shape)
        
        ball_position = observations['ball_position'][:,0]
        image = observations['masked_observation']

        #print(ball_position.shape)
        #print(image.shape)
        #
        ball_position = ball_position.view(-1, 2)
        x = self.cnn(image)

        return torch.cat((x, ball_position), dim=1)
    

def return_model(cfg, env, policy_kwargs=None):
    if cfg['model_name'] == 'ppo':
        log("Using PPO model")
        model = PPO(policy = cfg['policy_type'],
                    env = env,
                    learning_rate = cfg['learning_rate'],
                    tensorboard_log=results_dir,
                    policy_kwargs=policy_kwargs,  # POlicy kwargs
                    verbose=1)
        
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
    #logger.info(msg)
    print(msg, flush=True)
 

class TennisWrapper2(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)[0]
        self.current_score = [0,0]
        self.score = [0,0] 
        self.all_score = [0,0] 
        self.server = 0 
        self.frames_per_point = 0
        #self.run_length = 0

        obs_image = self.run_reset(obs['masked_observation'])
        obs = {'masked_observation': obs_image, 
               'ball_position': obs['ball_position']}
        return obs, {}

    def run_reset(self,old_obs,max_loop=1000):
        self.run_length = 0
        
        for _ in range(max_loop):
            obs, _, trunc, termi, _ = self.env.step(0)
            
            obs = obs['masked_observation']
            if not np.all(old_obs == obs):
                break
            old_obs = obs
            if trunc or termi:
                break
            
        self.run_length = 0
        return obs
        
    def step(self, action):
        #assert self.action_mask()[action], "not a legal action"

        if (self.server == 0 and self.run_length < (5 if self.current_score == [0,0] else 15)):
            action = 1
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.run_length += 1
        
        info["run_done"] = False
        if reward != 0:
            info["run_done"] = True
            info["run_length"] = self.run_length
            wandb.log({"run_length": self.run_length})
            
            #reward += info["run_length"] / 50
            run_winner = 0 if reward == 101.0 else 1
            self.current_score[run_winner] += 1
            self.all_score[run_winner] += 1
       
            if self.current_score[run_winner] >= 4 and self.current_score[run_winner] - self.current_score[1-run_winner]>=2:      
                self.current_score = [0,0]
                self.score[run_winner] += 1
                self.server = 1 - self.server
                
            obs_image = obs['masked_observation'] 
            
            if not (terminated or truncated):
                obs_image = self.run_reset(obs_image.copy())
            
            obs['masked_observation'] = obs_image
            self.run_length = 0
        #info["action_mask"] = self.action_mask()


        return obs, reward, terminated, truncated, info
    

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_mask()

if __name__ == "__main__":
   
    idx = time.strftime("%d%H%M%S")
    
    #name_run = f"TennisSB3_{idx}"
    name_run = "Tennis_A2CFinal_0"
    print("Overwriting the run name")
    run = wandb.init(project="Freeway", entity="pilligua2", name=name_run, sync_tensorboard=True, group="Tennis")

    results_dir = f"/ghome/mpilligua/RL/Project_RL/Tennis/runs/{name_run}"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/videos", exist_ok=True)
    
    
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

    ENV_NAME = "ALE/Tennis-v5"
    def env_make(do_eval=False):
        if do_eval:
            env = gym.make(ENV_NAME, render_mode='rgb_array')
        else:
            env = gym.make(ENV_NAME)
        env = MaxAndSkipObservation(env, 4)
        env = ResizeObservation(env, shape=(84, 84))
        env = ImageToPyTorch(env)  # Transpose observation to (C, H, 
        env = CropObservation(env, 5, 6, 80, 80)
        env = GetBallPosition(env)
        env = FrameStackObservation(env, stack_size=4)
        env = EndWhenLosePoint(env, do_eval=do_eval)
        env = MapActions(env)
        env = Monitor(env)
        
        env = TennisWrapper2(env)
        
        return env
    
    train_env = env_make(do_eval=False)
    #train_env = ActionMasker(train_env, mask_fn)
    
    eval_env = env_make(do_eval=True)
    #eval_env = ActionMasker(eval_env, mask_fn)

    # Callbacks
    eval_callback = CustomEvalCallback(eval_env, export_path = results_dir, eval_freq=10000, verbose=2)
    # callback = CallbackList([eval_callback, WandbCallback()])
#
    print('\n>>> Creating and training model ..')
#
    # Change here the type of model to train from the configs/SB3.py file
    training_config['model_arguments'] = ppo_arguments
    
    print(f"Training config: {training_config}")
    
    policy_kwargs = {
        'net_arch': dict(pi=[], vf=[]),
        'features_extractor_class': CustomCNN,
        'features_extractor_kwargs': {'features_dim': 34},
        'normalize_images': False
    }

    if training_config['number_of_actions'] == 'Reduce':
        log("Reducing the number of actions to 6")
        train_env.action_space.n = 6
        eval_env.action_space.n = 6
    
    
    
    wandb.config.update(training_config)
    
    model = A2C("MultiInputPolicy", 
                    env = train_env, 
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=results_dir)
        
        
    
    model.learn(total_timesteps=10000000, callback = eval_callback)

    model.save(os.path.join(results_dir, name_run))
    print(f"Model exported at '{os.path.join(results_dir, name_run)}'")
    run.finish()


