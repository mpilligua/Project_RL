from stable_baselines3 import PPO
import supersuit as ss
from pettingzoo.atari import pong_v3
import numpy as np
import gymnasium as gym
from pettingzoo.utils import aec_to_parallel
import sys

import os
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
from PIL import Image, ImageDraw, ImageFont
from math import floor

from collections import deque

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common import utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import time
from tqdm import tqdm

import cv2
import warnings
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from supersuit.aec_vector.async_vector_env import AsyncAECVectorEnv
from supersuit.utils.base_aec_wrapper import BaseWrapper

from supersuit.lambda_wrappers.observation_lambda import aec_observation_lambda
import cloudpickle

class SingleAgentEnv(gym.Env):
    """Wraps the multi-agent environment to allow a single agent to interact in a turn-based fashion."""
    def __init__(self, env, agent_name):
        self.env = env
        self.agent_name = agent_name
        # print(f"Action space: {self.env.action_space}")
        self.action_space = self.env.action_space(self.agent_name)
        # print(f"Action space: {self.action_space}")
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84, 4), dtype=np.float32)
        self.num_envs = 1

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        observations, _, _, _, infos = self.env.last()
        return observations, infos

    def step(self, action):
        # for i, agent in enumerate(env.agent_iter())
        print(f"Action: {action}")
        self.env.step(action)
        obs, rewards, terminated, truncated, infos = self.env.last()
        done = terminated or truncated
        return (
            np.array([obs]),
            np.array([rewards]),
            np.array([done]),
            [infos],
        )

class Monitor_pers(AsyncAECVectorEnv):
    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        envv: gym.Env,
        num_envs: int = 1,
        num_cpus: int = 1,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        def env_fn():
            # filtered_words=stopwords_delete(wordlist.flatMap(lambda x:x).collect())
            # print(filtered_words)
            return cloudpickle.loads(cloudpickle.dumps(envv))

        env_list = [env_fn] * num_envs

        super().__init__(env_list, num_envs, num_cpus)
        self.t_start = time.time()
        self.results_writer = None
        if filename is not None:
            env_id = env.spec.id if env.spec is not None else None
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": str(env_id)},
                extra_keys=reset_keywords + info_keywords,
                override_existing=override_existing,
            )

        self.inter_rewards = {}
        self.episode_returns = {}
        self.episode_lengths = {}
        self.episode_times = {}
        self.ep_rew = {}
        self.ep_len = {}
        self.ep_info = {}
        
        for agent in env.possible_agents:
            self.episode_returns[agent] = []
            self.episode_lengths[agent] = []
            self.episode_times[agent] = []
        self.total_steps = 0
        # extra info about the current episode, that was passed in during reset()
        self.current_reset_info: Dict[str, Any] = {}
        self.super = super()
        self.update = False
        
        self.last_reward = {}
        for agent in self.env.possible_agents:
            self.inter_rewards[agent] = [[] for _ in range(self.num_envs)]
            self.last_reward[agent] = np.array([0 for _ in range(self.num_envs)])
        
    def reset(self, **kwargs):
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        self.super.reset(**kwargs)  
        
        for agent in self.env.possible_agents:
            self.episode_returns[agent] = []
            self.episode_lengths[agent] = []
            self.episode_times[agent] = []
            
        # self.total_steps = 0
        self.inter_rewards = {}
        self.last_reward = {}
        for agent in self.env.possible_agents:
            self.inter_rewards[agent] = [[] for _ in range(self.num_envs)]
            self.last_reward[agent] = np.array([0 for _ in range(self.num_envs)])

    def step(self, action):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        
        self.super.step(action)        
        
        
    
    def last(self):
        agent = self._find_active_agent()
        
        observation, reward, terminated, truncated, info = self.super.last()
        # print(self.inter_rewards[agent])
        for i in range(self.num_envs):
            self.inter_rewards[agent][i].append(reward[i])
        
        if np.any(reward != self.last_reward[agent]):
            # print(reward)
            # print(self.last_reward[agent])
            # print(np.where(reward != self.last_reward[agent]))
            indx_diff = np.where(reward != self.last_reward[agent])[0][0]
            print(self.inter_rewards[agent][indx_diff])
            rew = self.inter_rewards[agent][indx_diff]
            ep_rew = sum(rew)
            ep_len = len(rew)
            # print(rew)
            # print(f"Reward changed from {reward[indx_diff]} to {ep_len}")
            # reward[indx_diff] = ep_len
            # ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
                
            # self.episode_returns[agent].append(ep_rew)
            # self.episode_lengths[agent].append(ep_len)
            # self.episode_times[agent].append(time.time() - self.t_start)
            # ep_info.update(self.current_reset_info)
            
            wandb.log({f"rollout/{agent}_mean_episode_reward": ep_rew})
            if ep_len > 1:
                wandb.log({f"rollout/mean_episode_length": ep_len})
            self.last_reward[agent] = reward
            self.inter_rewards[agent][indx_diff] = []
        self.total_steps += 1
        
          
        return observation, reward, terminated, truncated, info
            
            
        return 

# Create AEC environment (turn-based interaction)
def make_env(num_envs):
    env = pong_v3.env(render_mode="rgb_array")  # AEC environment
    possible_agents = env.possible_agents
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    if num_envs > 1:
        envv = ss.vectorize_aec_env_v0(env, num_envs=num_envs, num_cpus=4)
        # print(env)
        # exit(0)
        env = Monitor_pers(envv, env, num_envs=num_envs, num_cpus=4)
        # print(env)
    return env


class GetBallPosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_ball_position = (0, 0)
        self.maskara = cv2.imread('/ghome/mpilligua/RL/Project_RL/Tennis/MariaMaskara2.jpg')
        self.maskara = self.maskara.mean(axis=-1)
        _, self.maskara = cv2.threshold(self.maskara, 127, 255, cv2.THRESH_BINARY)
        #self.maskara = self.maskara[np.newaxis, :, :]
        self.maskara = np.repeat(self.maskara[np.newaxis, :, :], 3, axis=0)
        
        # Define the observation space as a Dict

        self.observation_space = gym.spaces.Dict({
            "ball_position": gym.spaces.Box(low=0, high=80, shape=(1, 2), dtype=np.float32),
            "masked_observation": gym.spaces.Box(
                low=0,
                high=1,
                shape=(
                    79,#env.observation_space.shape[1],
                    78#env.observation_space.shape[2],
                ),
                dtype=np.float32,
            ),
        })
    def observation(self, observation):
        # Apply the mask to the observation
        
        obs = observation * self.maskara  # Ensure maskara has the same shape as observation
        obs /= 255.0
        
    
        # Get image dimensions (assuming channel-first: (channels, height, width))
        channels, height, width = obs.shape

        # Reshape the image for easier processing (flatten to [num_pixels, 3])
        flattened_obs = obs.transpose(1, 2, 0).reshape(-1, 3)  # Convert to (height, width, channels) first

        # Define ball colors and field colors as numpy arrays for vectorized operations
        ball_colors = np.array([
            [137, 179, 156],  # Ball color 1
            [183, 205, 193],  # Ball color 2
        ])
        field_colors = np.array([
            [45, 126, 82],     # Field green
            [147, 127, 100],   # Field red
            [120, 128, 224],   # Field blue
            [147, 127, 100],   # Field brown
        ])

        # Calculate Euclidean distances from each pixel to ball and field colors
        ball_distances = np.min(np.linalg.norm(flattened_obs[:, None, :] - ball_colors[None, :, :], axis=2), axis=1)
        field_distances = np.min(np.linalg.norm(flattened_obs[:, None, :] - field_colors[None, :, :], axis=2), axis=1)

        # Determine which pixels are ball-like
        threshold = 50
        is_ball_pixel = (ball_distances < field_distances) & (ball_distances < threshold)

        # Find the coordinates of ball-like pixels
        ball_pixel_indices = np.where(is_ball_pixel)[0]
        if ball_pixel_indices.size == 0:
            # If no ball pixel is found, return the last known position
            ball_coordinates = self.last_ball_position
        else:
            # Convert the flattened index back to (y, x) coordinates
            ball_coordinates = np.unravel_index(ball_pixel_indices[0], (height, width))
            self.last_ball_position = ball_coordinates


        #print(ball_coordinates)
        # Afegir la red dim i pasar la imatge normalizada

        red_channel = obs[0]

        # Return the processed observation and the ball position
        return {
            "masked_observation": red_channel.astype(np.float32) / 255.0,
            "ball_position": np.array(ball_coordinates, dtype=np.float32).reshape(1, 2),
        }


class NatureCNN_pers(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[-1]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim-2), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if len(observations.shape) == 3:
            # We add a batch dimension
            observations = observations.unsqueeze(0)
        # print(f"Observations shape: {observations.shape}")
        observations = observations.permute(0, 3, 1, 2)
        area2search = observations[:, -1, 14:78, :]
        getLightPixels = torch.where(area2search > 0.58)
        # ball coordinated must have dim batch,2
        # getLightPixels is a tuple with 3 elements, batch, y, x and we must transform it to batch, 2 by doing the mean of the y and x within each element of the batch
        ball_coordinates = torch.ones((observations.shape[0], 2), device=area2search.device) * -1
        
        for batch in range(observations.shape[0]):
            # Mask for the current batch
            batch_mask = getLightPixels[0] == batch
            
            # Extract y and x coordinates for this batch
            y_in_batch = getLightPixels[1][batch_mask]
            x_in_batch = getLightPixels[2][batch_mask]
            
            if len(y_in_batch) > 0:  # Ensure there are valid pixels in the batch
                mean_y = y_in_batch.float().mean()
                mean_x = x_in_batch.float().mean()
                ball_coordinates[batch] = torch.tensor([mean_x, mean_y], device=area2search.device)
        # print(f"Ball coordinates: {ball_coordinates}")
        # plt.imsave("image.png", (area2search[-1, -1]*255.0).cpu().numpy().astype(np.uint8), cmap="gray")
        # print(getLightPixels)
        # exit(0)
        ball_coordinates[:, 1] = ball_coordinates[:, 1] + 13
        self.ball_coordinates = ball_coordinates
        
        out = self.cnn(observations.float())
        out = out.view(out.size(0), -1)
        return torch.cat([self.linear(out), ball_coordinates], dim=1)
        # return torch.cat([self.linear(self.cnn(observations)), ball_coordinates], dim=1)


def visualize_train(env, ppo_models, episode =100000, epsilon=0.0, save_path = ""):
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        env.reset()
        obs, _, _, _, _ = env.last()
    
        done = False
        images = []
        total_reward = 0.0
        t = 0
        const_text_height = 25
        reward_model1 = 0
        reward_model2 = 0
        action_model1 = 0
        action_model2 = 0
        probabs_model1 = torch.zeros(6)
        probabs_model2 = torch.zeros(6)
        colors = plt.get_cmap("viridis")

        for agent in env.agent_iter(2**63):    
            img = env.render()
            img = img.astype(np.uint8)
            img_pil = Image.fromarray(img)

            # Make a step in the environment
            action, _ = policies[agent].predict(obs)
            env.step(action)
            obs, reward, terminated, truncated, _ = env.last()
            ball_coords = policies[agent].policy.features_extractor.ball_coordinates[0].cpu()
            # print(ball_coords)
            
            
            img_array = np.expand_dims(obs, 0)
            img_array = np.repeat(img_array, 3, axis=0)
            img_array = np.moveaxis(img_array, 0, -1)  # Move the channel axis to the end
            img_array = (img_array * 255).astype(np.uint8)  # Convert to uint8
            img_array = img_array[:, :, -1]
            img_array[int(floor(ball_coords[1])), int(floor(ball_coords[0]))] = [255, 0, 0]
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
            
            step_text = f"Step: {t}"
            text_bbox = draw.textbbox((0, 0), step_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((2*(combined_image.width - text_width) // 3, 1), step_text, fill="white", font=font)
            
            # Add the epoch number centered at the top of the image
            epoch_text = f"Episode: {episode}"
            text_bbox = draw.textbbox((0, 0), epoch_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width), 1), epoch_text, fill="white", font=font)

            if agent == "first_0":
                action_model2 = action
                reward_model1 += reward
                # probabs_model2 = actions_prob
            else:
                action_model1 = action
                reward_model2 += reward
                # probabs_model1 = actions_prob
        
            reward_text = f"Reward: {reward_model1:.2f}"
            text_bbox = draw.textbbox((0, 0), reward_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((5, 10), reward_text, fill="white", font=font)
            
            # Add the action taken to the top-left corner of the image below the steps
            action_text = f"Act: {action_model1}"
            text_bbox = draw.textbbox((0, 0), action_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((5, 1), action_text, fill="white", font=font)

            # Add the current reward to the top-right corner of the image
            reward_text = f"Reward: {reward_model2:.2f}"
            text_bbox = draw.textbbox((0, 0), reward_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((img_pil.width - 5-text_width, 10), reward_text, fill="white", font=font)
            
            # Add the action taken to the top-left corner of the image below the steps
            action_text = f"Act: {action_model2}"
            text_bbox = draw.textbbox((0, 0), action_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((img_pil.width - 5-text_width, 1), action_text, fill="white", font=font)

            images.append(combined_image)
            
            if terminated or truncated or t > 3000:
                break

            t += 1

        # save the image
        images[0].save(save_path, save_all=True, append_images=images[1:], duration=20, loop=0)

        env.close()
        return total_reward
    

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

class PPO(OnPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        print(f"Batch size: {batch_size}")
        print(f"n_epochs: {n_epochs}")

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        
        # train for n_epochs epochs
        for epoch in tqdm(range(self.n_epochs), desc="Train", unit="epoch"):
            approx_kl_divs = []
            mean_reward = []
            # Do a complete pass on the rollout buffer
            # print(f"SBatch size: {self.batch_size} Buffer size: {rollout_buffer)}")
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
                mean_reward.append(rollout_data.returns.mean().item())
                

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        # self.logger.record("agent/agent", self.agent_name)
        self.logger.record(f"train/{self.agent_name}_entropy_loss", np.mean(entropy_losses))
        self.logger.record(f"train/{self.agent_name}_policy_gradient_loss", np.mean(pg_losses))
        self.logger.record(f"train/{self.agent_name}_value_loss", np.mean(value_losses))
        self.logger.record(f"train/{self.agent_name}_approx_kl", np.mean(approx_kl_divs))
        self.logger.record(f"train/{self.agent_name}_clip_fraction", np.mean(clip_fractions))
        self.logger.record(f"train/{self.agent_name}_loss", loss.item())
        self.logger.record(f"train/{self.agent_name}_explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(f"train/{self.agent_name}_std", th.exp(self.policy.log_std).mean().item())

        self.logger.record(f"train/{self.agent_name}_n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record(f"train/{self.agent_name}_clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record(f"train/{self.agent_name}_clip_range_vf", clip_range_vf)
        self.logger.record(f"train/{self.agent_name}_mean_reward", np.mean(mean_reward))

    

    def setup_learning(self, total_timesteps, callback, tb_log_name, reset_num_timesteps, progress_bar):
        """
        Perform initial setup for learning.
        """
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())
        assert self.env is not None, "Environment must be set before training."
        return total_timesteps, callback

    def collect_and_rollout(self, callback, n_steps):
        """
        Collect rollouts for training.
        """
        return self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=n_steps)

    def update_progress(self, iteration, log_interval, total_timesteps):
        """
        Update progress and log information.
        """
        self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

        if log_interval is not None and iteration % log_interval == 0:
            assert self.ep_info_buffer is not None
            self._dump_logs(iteration)

    def train_step(self):
        """
        Perform a single training step.
        """
        self.train()


    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        # Step 1: Setup Learning
        total_timesteps, callback = self.setup_learning(
            total_timesteps, callback, tb_log_name, reset_num_timesteps, progress_bar
        )

        # Step 2: Training Loop
        while self.num_timesteps < total_timesteps:
            # Step 2.1: Collect Rollouts
            continue_training = self.collect_and_rollout(callback, self.n_steps)

            if not continue_training:
                break

            iteration += 1

            # Step 2.2: Update Progress
            self.update_progress(iteration, log_interval, total_timesteps)

            # Step 2.3: Perform Training Step
            self.train_step()

        # Step 3: End Training
        callback.on_training_end()

        return self

    def prev_to_loop(self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,):
        
        self.total_timesteps, self.callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        
        self.callback.on_training_start(locals(), globals())

        # assert self.env is not None

        # self.obs = self.env.reset()  # Initialize environment
        # self._last_obs = obs
        # self.num_timesteps = 0
        
    def setup_loop_over_timesteps(self: SelfOnPolicyAlgorithm):
        # Manual collect_rollouts
        self.policy.set_training_mode(False)
        
        self.rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(self.env.num_envs)

        self.callback.on_rollout_start()

    def get_action(self: SelfOnPolicyAlgorithm, obs, n_steps):
        self._last_obs = obs
        if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.policy.reset_noise(self.env.num_envs)
    
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            actions, values, log_probs = self.policy(obs_tensor)

        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions

        if isinstance(self.action_space, spaces.Box):
            if self.policy.squash_output:
                # Unscale the actions to match env bounds
                # if they were previously squashed (scaled in [-1, 1])
                clipped_actions = self.policy.unscale_action(clipped_actions)
            else:
                # Otherwise, clip the actions to avoid out of bound error
                # as we are sampling from an unbounded Gaussian distribution
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return clipped_actions, values, log_probs

    def predict_last_seen(self: SelfOnPolicyAlgorithm, new_obs, dones):
        with th.no_grad():
            # Compute value for the last timestep
            # print(f"New obs: {new_obs.shape}")
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        self.callback.update_locals(locals())

        self.callback.on_rollout_end()


    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        # self.logger.record("agent/agent", self.agent_name)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(f"rollout/{self.agent_name}_ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record(f"rollout/{self.agent_name}_ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record(f"time/{self.agent_name}_fps", fps)
        self.logger.record(f"time/{self.agent_name}_time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record(f"rollout/{self.agent_name}_success_rate", safe_mean(self.ep_success_buffer))


    def update_buffer(self: SelfOnPolicyAlgorithm, new_obs, rewards, dones, infos, actions, values, log_probs, num_timesteps):
        self.num_timesteps = num_timesteps
        # Give access to local variables
        self.callback.update_locals(locals())
        if not self.callback.on_step():
            return False

        self._update_info_buffer(infos, dones)

        if isinstance(self.action_space, spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstrapping with value function
        # see GitHub issue #633
        for idx, done in enumerate(dones):
            if (
                done
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                rewards[idx] += self.gamma * terminal_value

        self.rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,)

        self._last_obs = new_obs  # type: ignore[assignment]
        self._last_episode_starts = dones


    def train_and_log(self: SelfOnPolicyAlgorithm, iteration, log_interval, total_timesteps, log_common=True):
        self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            assert self.ep_info_buffer is not None
            self._dump_logs(iteration)
            
        self.train()

    def _setup_learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run",
            progress_bar: bool = False,
        ) -> Tuple[int, BaseCallback]:
        
            self.start_time = time.time_ns()

            if self.ep_info_buffer is None or reset_num_timesteps:
                # Initialize buffers if they don't exist, or reinitialize if resetting counters
                self.ep_info_buffer = deque(maxlen=self._stats_window_size)
                self.ep_success_buffer = deque(maxlen=self._stats_window_size)

            if self.action_noise is not None:
                self.action_noise.reset()

            if reset_num_timesteps:
                self.num_timesteps = 0
                self._episode_num = 0
            else:
                # Make sure training timesteps are ahead of the internal counter
                total_timesteps += self.num_timesteps
            self._total_timesteps = total_timesteps
            self._num_timesteps_at_start = self.num_timesteps

            # Configure logger's outputs if no logger was passed
            if not self._custom_logger:
                self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

            # Create eval callback if needed
            callback = self._init_callback(callback, progress_bar)

            return total_timesteps, callback

    def manual_learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        A manual implementation of the learn function, allowing direct control of the environment.

        :param total_timesteps: Total number of timesteps to train for.
        :param callback: Optional callback for monitoring training progress.
        """
        
        iteration = 0
        self.prev_to_loop(total_timesteps, callback, tb_log_name, reset_num_timesteps, progress_bar)
        
        num_timesteps = 0
        while num_timesteps < total_timesteps:
            n_steps = 0
            self.setup_loop_over_timesteps()
                
            obs = self.env.reset()  # type: ignore[assignment]
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            
            while n_steps < self.n_steps:
                actions, values, log_probs = self.get_action(obs, n_steps)
                new_obs, rewards, dones, infos = self.env.step(actions)
        
                num_timesteps += self.env.num_envs  
                self.update_buffer(new_obs, rewards, dones, infos, actions, values, log_probs, num_timesteps)
                n_steps += 1
                
            self.predict_last_seen(obs, dones)

            iteration += 1
            
            self.train_and_log(iteration, log_interval, total_timesteps)
            
            
        self.callback.on_training_end()


class AggregatingLogger(Logger):
    def __init__(self, existing_logger):
        self.values = {}
        self.super = existing_logger
        self.level = self.super.level
        self.output_formats = self.super.output_formats
        self.name_to_value = self.super.name_to_value
        self.name_to_excluded = self.super.name_to_excluded
        self.name_to_count = self.super.name_to_count
        self.dir = self.super.dir
        
        # self.name_to_value: Dict[str, float] = defaultdict(float)  # values this iteration
        # self.name_to_excluded: Dict[str, Tuple[str, ...]] = {}
        # self.level = INFO
        # self.dir = folder
        # self.output_formats = output_formats
        

    def record(self, key, value, exclude="tensorboard"):
        self.values[key] = value
        self.super.record(key, value, exclude=exclude)

    def get_logs(self):
        return self.values


if __name__ == "__main__":
    policy_kwargs = {
        "features_extractor_class": NatureCNN_pers,
        "features_extractor_kwargs": {"features_dim": 32},
        'net_arch': dict(pi=[16], vf=[16]),
    }

    num_envs = 1
    env = make_env(num_envs)
    env.num_envs = num_envs
    env.reset()

    eval_env = make_env(1)
    eval_env.num_envs = 1 

    device = "cuda" if th.cuda.is_available() else "cpu"

    right_agent = PPO("CnnPolicy", SingleAgentEnv(env, "first_0"), verbose=1, policy_kwargs=policy_kwargs, device=device, n_epochs=5, 
                ent_coef = 0.5,
                gae_lambda = 0.65,)

    left_agent = PPO("CnnPolicy", SingleAgentEnv(env, "second_0"), verbose=1, policy_kwargs=policy_kwargs, device=device, n_epochs=5, 
                ent_coef = 0.5,
                gae_lambda = 0.65,)

    right_agent.policy.load_state_dict('Path_to_ppo_weight_right')
    left_agent.policy.load_state_dict('Path_to_ppo_weight_left')