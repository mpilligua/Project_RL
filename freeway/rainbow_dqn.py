import torch
from copy import deepcopy
import numpy as np
import wandb
import warnings
from torch import nn
import torch.optim as optim


from DQN_extensions.types_of_buffer import ExperienceReplay, PrioritizedExperienceReplayBuffer
from DQN_extensions.dqn_with_noise import Noisy_DQN

from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import gymnasium as gym

import collections

from PIL import Image

import ale_py
from PIL import ImageDraw, ImageFont
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

gym.register_envs(ale_py)

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

def make_env(env_name, skip=4, stack_size=4, reshape_size=(84, 84), render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    print("Standard Env.        : {}".format(env.observation_space.shape))
    env = MaxAndSkipObservation(env, skip=skip)
    print("MaxAndSkipObservation: {}".format(env.observation_space.shape))
    env = ResizeObservation(env, reshape_size)
    print("ResizeObservation    : {}".format(env.observation_space.shape))
    env = GrayscaleObservation(env, keep_dim=True)
    print("GrayscaleObservation : {}".format(env.observation_space.shape))
    env = ImageToPyTorch(env)
    print("ImageToPyTorch       : {}".format(env.observation_space.shape))
    env = ReshapeObservation(env, reshape_size)
    print("ReshapeObservation   : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=stack_size)
    print("FrameStackObservation: {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    
    return env

class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(DQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape)
        )
    
    def forward(self, x):
        return self.net(x)
        


def create_and_save_gif(net, env, device, save_gif='video.gif', epoch=0):  # CREDITS JORDI
    current_state = env.reset()[0]
    
    done = False
    images = []
    total_reward = 0.0
    t = 0
    
    while not done:
        img = env.render()
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil) 
        
        state_ = np.array([current_state])
        state = torch.tensor(state_).to(device)
        q_vals = net(state)
        action = np.argmax(q_vals)
        
        # Add the step number to the upper-left corner of the image
        font = ImageFont.load_default() 
        draw.text((10, 10), f"Step: {t}", fill="white", font=font)

        # Add the epoch number centered at the top of the image
        epoch_text = f"Episode: {epoch}"
        text_bbox = draw.textbbox((0, 0), epoch_text, font=font) 
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.text(((img_pil.width - text_width) // 2, 10), epoch_text, fill="white", font=font)
    
        # Add the current reward to the top-right corner of the image
        total_reward += reward
        reward_text = f"Reward: {total_reward:.2f}"
        text_bbox = draw.textbbox((0, 0), reward_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.text((img_pil.width - text_width - 10, 10), reward_text, fill="white", font=font)
        
        current_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated    
        total_reward += reward  
        images.append(img_pil)
        
        t += 1
        
    images[0].save(save_gif, save_all=True, append_images=images[1:], duration=100, loop=0)
    
    env.close()
    return total_reward

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'combined_reward', 'combined_done', 'next_new_state'])

class RainbowDQN_Agent:
    """
    A Rainbow DQN Agent that incorporates various extensions to the standard DQN algorithm,
    including Double DQN, Dueling Networks, Prioritized Experience Replay, and Noisy Networks.
    This agent is designed to interact with a Gymnasium environment, learn optimal policies,
    and track performance metrics over time.
    """

    def __init__(self, 
                 train_env: gym.Env,
                 eval_env: gym.Env, 
                 config: dict,
                 device: torch.device,
                 results_dir: str = None):
        """
        Initialize the Rainbow DQN Agent.

        Args:
            env (gym.Env): The Gymnasium environment the agent will interact with.
            dnnetwork (nn.Module): The neural network model used for approximating Q-values.
            buffer (ExperienceReplay): The experience replay buffer for storing past experiences.
            batch_size (int, optional): Number of samples per training batch. Defaults to 32.
            epsilon (float, optional): Initial epsilon value for the epsilon-greedy policy. Defaults to 1.0.
            eps_decay (float, optional): Decay rate for epsilon after each step. Defaults to 0.999985.
            use_double_dqn (bool, optional): Whether to use Double DQN for reducing overestimation bias. Defaults to False.
            use_two_step_dqn (bool, optional): Whether to use Two-Step DQN for better credit assignment. Defaults to False.
            use_dueling_dqn (bool, optional): Whether to use Dueling DQN architecture. Defaults to False.
            use_noise_dqn (bool, optional): Whether to use Noisy Networks for exploration. Defaults to False.
            reward_threshold (float, optional): The average reward threshold to consider the environment as solved. Defaults to 32.0.
        """
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env
        
        self.dqn_extensions = config['dqn_extensions']
        
        self.buffer = setup_buffer(self.dqn_extensions['use_prioritized_buffer'], config)
        
        if self.dqn_extensions['use_noisy_dqn']:
            # Ensure consistent initialization by passing necessary parameters
            dnnetwork = Noisy_DQN(self.train_env.observation_space.shape, self.train_env.action_space.n, noisy_std = 0.1, device=device).to(device)
        else:
            dnnetwork = DQN(self.train_env.observation_space.shape, self.train_env.action_space.n).to(device)
        
        # Networks
        self.dnnetwork = dnnetwork.to(device)
        self.target_network = deepcopy(dnnetwork).to(device)
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        self.batch_size = config['batch_size']
        
        # Optimizer for training the network
        self.optimizer = optim.Adam(self.dnnetwork.parameters(), lr=config['learning_rate'])
        
        # Flags to determine which DQN extensions to use
        self.double_dqn = dqn_extensions['use_double_dqn']
        self.two_step_dqn = dqn_extensions['use_two_step_dqn']
        self.dueling_dqn = dqn_extensions['use_dueling_dqn']
        self.use_noise_dqn = dqn_extensions['use_noisy_dqn']
        
        # Metrics for tracking performance
        self.total_rewards = []
        self.number_of_frames_Episodes = []
        self.update_loss = []
        
        # Exploration parameters for epsilon-greedy strategy
        self.epsilon = config["eps_start"]
        self.eps_decay = config["eps_decay"]
        self.eps_min = config["eps_min"]
        
        self.gamma=config["gamma"]
        self.max_frames = config["max_frames"]
        self.dnn_upd_freq = config["dnn_upd"]
        self.dnn_sync_freq = config["dnn_sync"]
        
        # Frame counter to keep track of the number of steps taken
        self.frame_number = 0
        
        # Check if the replay buffer uses prioritized experience replay
        self.prioritized_buffer = dqn_extensions['use_prioritized_buffer']
        self.results_dir = results_dir
        
        # Reset any noise in the network if using Noisy DQN
        self.reset_noise()
        
        # Initialize variables specific to each episode
        self._reset()
        
    # STEP FROM 2-STEP DQN
    def take_step(self, epsilon=0.0, mode='explore', device="cpu"):
        done_reward = None
        done_number_steps = None
        
        def get_action(state):
            if mode == 'explore':
                return self.env.action_space.sample()
            elif np.random.random() < epsilon:
                return self.env.action_space.sample()
            else:
                state_ = np.array([state])
                state_tensor = torch.tensor(state_).to(device)
                q_vals = self.dnnetwork(state_tensor)
                _, act_ = torch.max(q_vals, dim=1)
                return int(act_.item())
        
        # First Step
        action = get_action(self.current_state)
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward
        
        self.number_of_frames_per_episode += 1
        
        if not self.two_step_dqn:  # If we are not using two step DQN
            self.buffer.append(self.current_state, action, reward, is_done, new_state)
            
            self.current_state = new_state
            if is_done:
                done_reward = self.total_reward
                done_number_steps = self.number_of_frames_per_episode
                self._reset()
            
            return done_reward, done_number_steps
    
        if not is_done:  # Only take a second step if the first is not terminal
            next_action = get_action(new_state)
            next_new_state, next_reward, terminated, truncated, _ = self.env.step(next_action)
            next_done = terminated or truncated
            self.number_of_frames_per_episode += 1
        else:
            next_new_state = new_state
            next_reward = 0
            next_done = True
    
        combined_reward = reward + self.gamma * next_reward
        combined_done = is_done or next_done
    
        self.buffer.append(self.current_state, action, combined_reward, combined_done, next_new_state)
        
        # Update the current state
        self.current_state = next_new_state
        
        if is_done:
            done_reward = self.total_reward
            done_number_steps = self.number_of_frames_per_episode
            self._reset()
        
        return done_reward, done_number_steps

    def train(self):
        logger.info("Filling replay buffer...")

        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore', device=device)
            self.frame_number += 1
            if self.use_noise_dqn and (self.frame_number % self.noise_upd_freq == 0):
                self.reset_noise()
            
        episode = 0
        logger.info("Training...")

        while True:
            self.frame_number +=1
            
            done_reward, done_number_steps = self.take_step(self.epsilon, mode='train', device=device)

            if self.frame_number % self.dnn_upd_freq == 0:
                    self.update(device)
                    
            if self.frame_number % self.dnn_sync_freq == 0:
                self.target_network.load_state_dict(self.dnnetwork.state_dict())
                
            if done_reward is not None:
                # If the episode is done, log the results
                self.total_rewards.append(done_reward)
                self.number_of_frames_Episodes.append(done_number_steps)
                mean_reward = np.mean(self.total_rewards[-NUMBER_OF_REWARDS_TO_AVERAGE:])
                
                logger.info(f"Frame:{self.frame_number} | Total games:{len(self.total_rewards)} | Mean reward: {mean_reward:.3f}  (epsilon used: {self.epsilon:.2f})")
                
                wandb.log({"epsilon": self.epsilon, "reward_100": mean_reward, "reward": done_reward,
                            "Frames per episode": done_number_steps}, step=self.frame_number)
        
                episode += 1
            
                if self.frame_number >= self.max_frames:
                    logger.info('\nFrame limit reached.')
                    break

                # The game ends if the average reward has reached the threshold
                if mean_reward >= self.train_env.reward_threshold:
                    logger.info('\nEnvironment solved in {} episodes!'.format(episode))
                    break
                
                if self.episode % 100 == 0:
                    logger.info("Visualizing the model...")
                    create_and_save_gif(self.dnnetwork, 
                                        eval_env,
                                        device=device, 
                                        save_gif=f"{self.results_dir}/frame_{str(self.episode)}.gif",
                                        epoch=episode)
                
            if self.epsilon is not None:         
                self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
                   
    def calculate_loss(self, batch, weights, idxs, device="cpu"):
        states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
        rewards = torch.FloatTensor(rewards).to(device)
        actions = torch.LongTensor(actions).view(-1, 1).to(device)
        dones = torch.BoolTensor(dones).to(device)
        weights_tensor = torch.FloatTensor(weights).to(device)
        
        qvals = self.dnnetwork.get_qvals(states).gather(1, actions)
        
        if self.double_dqn:
            actions_next = torch.argmax(self.dnnetwork.get_qvals(next_states), dim=1).view(-1, 1)
            q_vals_next = self.target_network.get_qvals(next_states)
            qvals_next = q_vals_next.gather(1, actions_next)
        else:
            qvals_next = torch.max(self.target_network.get_qvals(next_states), dim=1)[0].detach()
                
        qvals_next[dones] = 0
        
        expected_qvals = rewards + self.gamma * qvals_next

        # Loss for the prioritized buffer
        if self.prioritized_buffer:
            deltas = expected_qvals.unsqueeze(1) - qvals
            priorities = deltas.abs().cpu().detach().numpy().flatten()
            self.buffer.update_priorities(idxs, priorities + 1e-6)
            loss = (deltas**2 * weights_tensor.view(-1, 1)).mean()
        else:
            loss = self.mse_loss(qvals, expected_qvals.unsqueeze(1))
        
        return loss

    def update(self, device):
        self.dnnetwork.zero_grad()
        idxs, batch, normalized_weight = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch, normalized_weight, idxs, device)
        loss.backward()
        self.optimizer.step()
        
        self.update_loss.append(loss.item())
        
    def reset_noise(self):
        for module in self.dnnetwork.modules():
            if hasattr(module, 'reset_noise'):
                module.reset_noise()
    
    def _reset(self):
        self.current_state = self.env.reset()[0]
        self.total_reward = 0
        self.number_of_frames_per_episode = 0


def setup_buffer(use_prioritized_buffer, config):
    EXPERIENCE_REPLAY_SIZE = config['experience_replay_size']
    config_prioritized_buffer = config['prioritized_buffer_config']
    
    if use_prioritized_buffer:
        buffer = PrioritizedExperienceReplayBuffer(
            experience_replay_size=EXPERIENCE_REPLAY_SIZE,
            alpha=config_prioritized_buffer['alpha'],
            small_constant=config_prioritized_buffer['small_constant'],
            growth_rate=config_prioritized_buffer['growth_rate'],
            beta=config_prioritized_buffer['beta']
        )
    else:
        buffer = ExperienceReplay(capacity=EXPERIENCE_REPLAY_SIZE)
    return buffer

if __name__ == "__main__":
    import os
    import time
    from configs.Rainbow_DQN import * # <--- All hyperparameters are defined here
    import logging
    
    idx = time.strftime("%d%H%M")
    name_run = f"freeway_rainbow_{idx}"
    
    wandb.init(project="Freeway", name=name_run)
    results_dir = f"/fhome/pmlai10/Project_RL/freeway/{name_run}"

    os.makedirs(results_dir, exist_ok=True)
    print("Saving to: ", results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # warnings.filterwarnings('ignore')

    train_env = make_env(**train_env_config)
    eval_env = make_env(**eval_env_config)
    
    # Initialize the agent
    agent = RainbowDQN_Agent(
        train_env=train_env,
        eval_env=eval_env, 
        config=training_config,
        device=device
    )

    wandb.config.update(all_configs)

    print("Training the agent...", flush=True)
    agent.train()
#