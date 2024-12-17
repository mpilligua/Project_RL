import gymnasium as gym
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import torch.nn.functional as F
import ale_py
from PIL import Image, ImageDraw, ImageFont
import seaborn
import os
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

gym.register_envs(ale_py)

def visualize_train(env, ppo_model, episode =100000, epsilon=0.0, save_path = ""):
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        current_state = env.reset()
    
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
            action, _ = ppo_model.predict(current_state, deterministic=True)
            # print(action)
            current_state, reward, truncated, _ = env.step(action)            
            
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
        
            # Add the epsilon between the epoch and the reward
            epsilon_text = f"Eps: {epsilon:.2f}"
            text_bbox = draw.textbbox((0, 0), epsilon_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((((combined_image.width - text_width) // 4)*3, 1), epsilon_text, fill="white", font=font)

            # Add the current reward to the top-right corner of the image
            # print(reward[0])
            total_reward += reward[0]
            # print(total_reward)
            reward_text = f"Reward: {total_reward:.2f}"
            text_bbox = draw.textbbox((0, 0), reward_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((combined_image.width - text_width - 10, 1), reward_text, fill="white", font=font)

            # Add the action taken to the top-left corner of the image below the steps
            action_text = f"Act: {action[0]}"
            text_bbox = draw.textbbox((0, 0), action_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width) // 4, 1), action_text, fill="white", font=font)

            done = truncated    
            # total_reward += reward  

            images.append(combined_image)

            t += 1

        # save the image
        images[0].save(save_path, save_all=True, append_images=images[1:], duration=100, loop=0)

        env.close()
        return total_reward
    
    


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
        self.super._on_step()
        # Evaluate the model every `eval_freq` steps
        if self.n_calls % self.eval_freq == 0:
            print(f"Step {self.n_calls}: Evaluating the model...")
            save_path = f"{self.save_path}/ppo_pong_{self.n_calls}.gif"
            visualize_train(self.eval_env, self.model, episode=self.n_calls, epsilon=0.0, save_path=save_path)
            wandb.log({"video":wandb.Image(save_path, caption=f"Episode {self.n_calls}")})
            # model_save_path = os.path.join(self.save_path, f"ppo_pong_{self.n_calls}.zip")
            # self.model.save(model_save_path)
            # print(f"Model saved at step {self.n_calls}")
            
        if self.n_calls % self.save_freq == 0:
            model.save(f"{save_path}/ppo_pong.zip")
        return True


class PongActionWrapper(gym.ActionWrapper):
   def action(self, act):
       # Map [0, 1, 2] to [NOOP, LEFT, RIGHT]
       mapping = {0: 0, 1: 3, 2: 2}
       return mapping[int(act)]


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
            nn.Conv2d(64, features_dim, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations = observations.permute(0, 2, 3, 1)
        # print(observations.shape)
        return self.cnn(observations)


# Check for CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

n_envs = 8  # Number of parallel environments
env = make_atari_env('PongNoFrameskip-v4', n_envs=n_envs, seed=0)
eval_env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
# env = PongActionWrapper(env)
# env.action_space = gym.spaces.Discrete(3)
# env = Monitor(env)

# Step 2: Apply frame stacking (important for Atari games)
env = VecFrameStack(env, n_stack=4)
eval_env = VecFrameStack(eval_env, n_stack=4)

# Load the base environment
# env = gym.make('ALE/Pong-v5', render_mode='rgb_array')  

# Apply the specified wrappers
# env = ss.color_reduction_v0(env, mode='B')  # Reduces the color of frames to black and white
# env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize the observation space
# env = ss.frame_stack_v1(env, 4)  # Stack 4 frames together
# env = ss.dtype_v0(env, dtype='float32')  # Change the data type of observations
# env = ss.normalize_obs_v0(env, env_min=0, env_max=1)  # Normalize the observation space


# Because the environment is now a non-vectorized environment after applying wrappers,
# we have to vectorize it using DummyVecEnv
# env = DummyVecEnv([lambda: env])

# Define the custom policy kwargs with the custom CNN and device
policy_kwargs = {
    'features_extractor_class': CustomCNN,
    'features_extractor_kwargs': {'features_dim': 3},
    'normalize_images': True
}

# Instantiate the agent on the specified device
name_run = "PongPPO5"
save_path = "Pong/runs/" + name_run
model = PPO("CnnPolicy", env, device=device, tensorboard_log=save_path, verbose=1)
os.makedirs(save_path, exist_ok=True)
print(f"Saving models to {save_path}")
run = wandb.init(project="Freeway", entity="pilligua2", name=name_run, sync_tensorboard=True)

eval_callback = EvalCallback(
    eval_env=eval_env,  # Use the same environment for evaluation
    eval_freq=50000,  # Evaluate every 50,000 steps
    save_path=save_path  # Path to save models
)

# Train the agent with a single call to `learn`
model.learn(total_timesteps=300000, callback=eval_callback)

model.save(f"{save_path}/ppo_pong.zip")
# model = PPO.load("Pong/ppo_pong.zip", env=env)

wandb.finish()