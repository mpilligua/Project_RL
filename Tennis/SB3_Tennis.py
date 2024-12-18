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


import seaborn
from wrappers import *
    

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


class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, export_path=None, eval_freq=1, verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval = 0
        self.export_path = export_path
        
    def _on_step(self) -> bool:
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

        while not done and t < 1000:
            img = self.eval_env.render()
            img = img[4:, 8:]
            img_pil = Image.fromarray(img)

            #action, q_values = self.get_action(current_state, mode='train', epsilon=epsilon, return_q_values=True)
            action = self.model.predict(current_state)[0]
            #current_state, reward, added_reward, terminated, truncated,  = self.eval_env.step(action)
            img_array = current_state[:, :, 0]
            #img_array = np.expand_dims(current_state[0], 0)
            #img_array = np.repeat(img_array, 3, axis=0)
            #img_array = np.moveaxis(img_array, 0, -1)  # Move the channel axis to the end
            img_array = (img_array * 255).astype(np.uint8)  # Convert to uint8
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
        images[0].save(f"{self.export_path}/videos/{self.n_calls}_.gif", save_all=True, append_images=images[1:], duration=50, loop=0)

        self.eval_env.close()
        return total_reward, 0

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
        wandb.log({"eval_reward": total_reward})
        images[0].save(save_gif, save_all=True, append_images=images[1:], duration=100, loop=0) 
        return float(total_reward), 0.0

    
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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


def return_model(cfg, env, policy_kwargs=None):
    if cfg['model_name'] == 'ppo':
        log("Using PPO model")
        model = PPO(policy = cfg['policy_type'],
                    env = env,
                    learning_rate = cfg['learning_rate'],
                
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
    
class KeepRedChannelWrapper(gym.ObservationWrapper):
    """
    A wrapper to keep only the red channel from RGB frames and resize them
    to the specified dimensions (default: 84x84).
    
    :param env: Gym environment to wrap
    :param width: Desired frame width after resizing
    :param height: Desired frame height after resizing
    """
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        
        # Ensure the input observation space is RGB
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"
        assert len(env.observation_space.shape) == 3 and env.observation_space.shape[2] == 3, \
            f"Expected RGB images with 3 channels, got shape {env.observation_space.shape}"
        
        # Update the observation space to reflect keeping only the red channel
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),  # Single channel for red
            dtype=np.uint8,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        Process the frame by extracting the red channel and resizing.

        :param frame: Input RGB frame from the environment
        :return: Processed frame with only the red channel and resized
        """
        assert cv2 is not None, "OpenCV is required. Install it with `pip install opencv-python`."

        # Extract the red channel (assuming frame is in RGB format)
        red_channel = frame[:, :, 0]

        # Resize the frame to the target dimensions
        resized_frame = cv2.resize(red_channel, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # Add a singleton dimension to make the output (height, width, 1)
        return resized_frame[:, :, None]


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.uint8) / 255



if __name__ == "__main__":
    import os
    import time
    from configs.SB3 import list_of_wrappers, sac_arguments, ac2_arguments, training_config, ppo_arguments, train_env_config, eval_env_config# <--- All hyperparameters are defined here
    import logging
    
    idx = time.strftime("%d%H%M%S")
    
    #name_run = f"TennisSB3_{idx}"
    name_run = "Probes"
    print("Overwriting the run name")
    
    run = wandb.init(project="Freeway", name=name_run, entity="pilligua2", group="Tennis", mode="disabled")
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

    #log(f"Initializing train environment with config: {train_env_config}\n")
    #train_env = make_env_sb3(train_env_config, evaluation=False)  # Ensure render_mode is passed as 'rgb_array'
#
    #log(f"\nInitializing eval environment with config: {eval_env_config}")
    #eval_env = make_env_sb3(train_env_config, evaluation=False)  # Separate evaluation environment

    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecFrameStack
    from stable_baselines3.common.atari_wrappers import FireResetEnv
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, ReshapeObservation
    from stable_baselines3.common.atari_wrappers import *
    from wrappers import *
    ENV_NAME = "ALE/Tennis-v5"
    def env_make(eval=False):
        if eval:
            env = gym.make(ENV_NAME, render_mode='rgb_array')
        else:
            env = gym.make(ENV_NAME)
       # type: ignore[attr-defined]

        env = FireResetEnv(env)
        log("FireResetEnv")
        if list_of_wrappers['ResizeObservation']:
            env = ResizeObservation(env, shape=(84, 84))

        if list_of_wrappers['VecTransposeImage']:
            env = ImageToPyTorch(env)  # Transpose observation to (C, H, W)

        if list_of_wrappers['clip_reward']:
            env = ClipRewardEnv(env)
            log("ClipRewardEnv")
        
        if list_of_wrappers['CropObservation']:
            env = CropObservation(env, 5, 6, 80, 80)
            log("CropObservation")

        #if list_of_wrappers['ReshapeObservation']:
        #    env = ReshapeObservation(env, shape=(84, 84))

        if list_of_wrappers['redChannel']:
            env = KeepRedChannelWrapper(env)
            log("KeepRedChannelWrapper")
            
        if list_of_wrappers['ScaledFloatFrame']:
            env = ScaledFloatFrame(env)
            log("ScaledFloatFrame")
        
        if list_of_wrappers['GetBallPosition']:
            env = GetBallPosition(env)
            log("GetBallPosition")

        return env
    
    train_env = make_vec_env(env_make, n_envs=8)
    eval_env = env_make(eval=True)

    if list_of_wrappers['VecFrameStack']:
        train_env = VecFrameStack(train_env, n_stack=4)
        log("VecFrameStack")


    # Callbacks
    eval_callback = CustomEvalCallback(eval_env, export_path = results_dir, eval_freq=training_config['eval_freq'], verbose=2)
    callback = CallbackList([eval_callback, WandbCallback()])
#
    print('\n>>> Creating and training model ..')
#
    # Change here the type of model to train from the configs/SB3.py file
    training_config['model_arguments'] = ppo_arguments
    
    print(f"Training config: {training_config}")
    
    policy_kwargs = dict()
    #policy_kwargs['normalize_images']=False
    if list_of_wrappers['ScaledFloatFrame']:
        policy_kwargs = policy_kwargs['normalize_images']=False

    if training_config['custom_dqn']:
        policy_kwargs['features_extractor_class'] = DQN_NET
        policy_kwargs['features_extractor_kwargs'] = dict(features_dim=40)
        

    if training_config['number_of_actions'] == 'Reduce':
        log("Reducing the number of actions to 6")
        train_env.action_space.n = 6
        eval_env.action_space.n = 6
    
    wandb.config.update(training_config)
    model = return_model(training_config['model_arguments'], train_env, policy_kwargs=policy_kwargs)
    
    model.learn(total_timesteps=training_config["total_timesteps"], callback = callback)

    model.save(os.path.join(results_dir, name_run))
    print(f"Model exported at '{os.path.join(results_dir, name_run)}'")
    run.finish()


