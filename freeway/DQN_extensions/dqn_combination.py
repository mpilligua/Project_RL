import torch
from copy import deepcopy
import numpy as np
import wandb
import warnings
from torch import nn
import torch.optim as optim


from DQN_extensions.types_of_buffer import ExperienceReplay, PrioritizedExperienceReplayBuffer
from DQN_extensions.dqn_with_noise import Noisy_DQN

from DQN_ex

from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import gymnasium as gym

import collections

from PIL import Image


import ale_py

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


def make_DQN(input_shape, output_shape):
    net = nn.Sequential(
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
    return net

def create_and_save_gif(net, device, save_gif='video.gif'):  # CREDITS JORDI
    env = make_env(ENV_NAME, render_mode='rgb_array')
    current_state = env.reset()[0]
    print(current_state)
    is_done = False
    
    images = []
    visualize = True
    total_reward = 0.0
    while not is_done:
        if visualize:
            img = env.render()
            images.append(Image.fromarray(img))
            
        state_ = np.array([current_state])
        state = torch.tensor(state_).to(device)
        q_vals = net(state)
        _, act_ = torch.max(q_vals, dim=1)
        action = int(act_.item())

        current_state, reward, terminated, truncated, _ = env.step(action)
        is_done = terminated or truncated    
        total_reward += reward  
    
    print("Total reward: %.2f" % total_reward)
    images[0].save(save_gif, save_all=True, append_images=images[1:], duration=100, loop=0)

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'combined_reward', 'combined_done', 'next_new_state'])

class RainbowDQN_Agent:
    """
    A Rainbow DQN Agent that incorporates various extensions to the standard DQN algorithm,
    including Double DQN, Dueling Networks, Prioritized Experience Replay, and Noisy Networks.
    This agent is designed to interact with a Gymnasium environment, learn optimal policies,
    and track performance metrics over time.
    """

    def __init__(self, 
                 env: gym.Env, 
                 dnnetwork: nn.Module, 
                 buffer: ExperienceReplay,
                 batch_size: int = 32,
                 epsilon: float = 1.0,
                 eps_decay: float = 0.999985,
                 use_double_dqn: bool = False,
                 use_two_step_dqn: bool = False,
                 use_dueling_dqn: bool = False,
                 use_noise_dqn: bool = False,
                 reward_threshold: float = 32.0):
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
        self.env = env
        # Networks
        self.dnnetwork = dnnetwork.to(device)
        self.target_network = deepcopy(dnnetwork).to(device)
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # Experience replay buffer
        self.buffer = buffer
        self.batch_size = batch_size
        
        # Optimizer for training the network
        self.optimizer = optim.Adam(self.dnnetwork.parameters(), lr=LEARNING_RATE)

        # Initialize the current state by resetting the environment
        self.current_state = self.env.reset()[0]
        
        # Reward threshold to determine when the environment is solved
        self.reward_threshold = reward_threshold
        
        # Flags to determine which DQN extensions to use
        self.double_dqn = use_double_dqn
        self.two_step_dqn = use_two_step_dqn
        self.dueling_dqn = use_dueling_dqn
        self.use_noise_dqn = use_noise_dqn
        
        # Metrics for tracking performance
        self.total_rewards = []
        self.number_of_frames_Episodes = []
        self.update_loss = []
        
        # Exploration parameters for epsilon-greedy strategy
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        
        # Frame counter to keep track of the number of steps taken
        self.frame_number = 0
        
        # Check if the replay buffer uses prioritized experience replay
        self.prioritized_buffer = isinstance(buffer, PrioritizedExperienceReplayBuffer)
        
        # Reset any noise in the network if using Noisy DQN
        self.reset_noise()
        
        # Initialize variables specific to each episode
        self._reset()
        
    # STEP FROM 2-STEP DQN
    def take_step(self, epsilon=0.0, mode='explore', device="cpu"):
        self.frame_number += 1
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
            exp = Experience(self.current_state, action, reward, is_done, new_state)
            self.buffer.append(exp)
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

    def train(self, gamma=0.99, 
              max_frames=50000,
              dnn_update_frequency=4, 
              dnn_sync_frequency=2000):
        
        self.gamma = gamma
        print("Filling replay buffer...")

        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

            if self.use_noise_dqn and self.frame_number % NOISE_UPD == 0:
                self.reset_noise()
            
        episode = 0
        training = True
        print("Training...")

        while training:
            self.current_state = self.env.reset()[0]
            self.total_reward = 0
            done_reward = None
            while done_reward is None:
                done_reward, done_number_steps = self.take_step(self.epsilon, mode='train')
                
                if self.frame_number % dnn_update_frequency == 0:
                    self.update()
                    
                if self.frame_number % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(self.dnnetwork.state_dict())
                    
                if done_reward is not None:
                    # If the episode is done, log the results
                    self.total_rewards.append(done_reward)
                    self.number_of_frames_Episodes.append(done_number_steps)
                    mean_reward = np.mean(self.total_rewards[-NUMBER_OF_REWARDS_TO_AVERAGE:])
                    print(f"Frame:{self.frame_number} | Total games:{len(self.total_rewards)} | Mean reward: {mean_reward:.3f}  (epsilon used: {self.epsilon:.2f})")
                    wandb.log({"epsilon": self.epsilon, "reward_100": mean_reward, "reward": done_reward,
                               "Frames per episode": done_number_steps}, step=self.frame_number)
            
                    episode += 1
                
                    if self.frame_number >= max_frames:
                        training = False
                        print('\nFrame limit reached.')
                        break

                    # The game ends if the average reward has reached the threshold
                    if mean_reward >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(episode))
                        break
                    
                    if self.frame_number % 10000 == 0:
                        create_and_save_gif(self.dnnetwork, 
                                            device=self.dnnetwork.device, 
                                            save_gif=f"/fhome/pmlai10/Project_RL/freeway/RainbowDQN_{idx}/frame_" + str(self.frame_number) + ".gif")
                        
                    self.epsilon = max(self.epsilon * self.eps_decay, EPS_MIN)

    def calculate_loss(self, batch, weights, idxs):
        states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
        rewards = torch.FloatTensor(rewards).to(self.dnnetwork.device)
        actions = torch.LongTensor(actions).view(-1, 1).to(self.dnnetwork.device)
        dones = torch.BoolTensor(dones).to(self.dnnetwork.device)
        weights_tensor = torch.FloatTensor(weights).to(self.dnnetwork.device)
        
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

    def update(self):
        self.dnnetwork.zero_grad()
        idxs, batch, normalized_weight = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch, normalized_weight, idxs)
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

# Start a new wandb run to track this script
import os
idx = len(os.listdir("/fhome/pmlai10/Project_RL/freeway"))
wandb.init(project="Freeway", name=f"freeway_{idx}")
os.makedirs(f"/fhome/pmlai10/Project_RL/freeway/RainbowDQN_{idx}", exist_ok=True)

print("Saving to: ", f"/fhome/pmlai10/Project_RL/freeway/RainbowDQN_{idx}")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

warnings.filterwarnings('ignore')

ENV_NAME = "ALE/Freeway-v5"
MEAN_REWARD_BOUND = 32
NUMBER_OF_REWARDS_TO_AVERAGE = 100          

GAMMA = 0.99   
    
BATCH_SIZE = 32  
LEARNING_RATE = 1e-4           

EXPERIENCE_REPLAY_SIZE = 10000            
SYNC_TARGET_NETWORK = 1000     

EPS_START = 1.0
EPS_DECAY = 0.999985
EPS_MIN = 0.02

DNN_UPD = 5
NOISE_UPD = 11000
DNN_SYNC = 1000
MAX_FRAMES = 1000000

use_noisy_dqn = True
use_double_dqn = True
use_two_step_dqn = True
use_dueling_dqn = True
use_prioritized_buffer = True

if use_prioritized_buffer:
    buffer = PrioritizedExperienceReplayBuffer(
        memory_size=EXPERIENCE_REPLAY_SIZE,
        burn_in=1000,
        alpha=0.6,
        small_constant=0.05,
        growth_rate=1.0005,
        beta=0.4
    )
else:
    buffer = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)

env = make_env(ENV_NAME)

if use_noisy_dqn:
    # Ensure consistent initialization by passing necessary parameters
    net = Noisy_DQN(env.observation_space.shape, env.action_space.n, noisy_std = 0.1).to(device)
else:
    net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)

# Remove target_net initialization from main script as it's handled within the agent

# Initialize the agent
agent = RainbowDQN_Agent(
    env=env, 
    dnnetwork=net,
    buffer=buffer,
    epsilon=EPS_START,
    eps_decay=EPS_DECAY,
    batch_size=BATCH_SIZE,
    use_double_dqn=use_double_dqn,
    use_two_step_dqn=use_two_step_dqn,
    use_dueling_dqn=use_dueling_dqn,
    use_noise_dqn=use_noisy_dqn,
    reward_threshold=MEAN_REWARD_BOUND  # Pass the reward threshold
)

env_config = {
    "skip": 4,
    "stack_size": 4,
    "reshape_size": (84, 84),
    "render_mode": None,
    "env_name": ENV_NAME,
    "use_noisy_dqn": use_noisy_dqn,
    "use_double_dqn": use_double_dqn,
    "use_two_step_dqn": use_two_step_dqn,
    "use_dueling_dqn": use_dueling_dqn,
    "use_prioritized_buffer": use_prioritized_buffer,
    "gamma": GAMMA,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "experience_replay_size": EXPERIENCE_REPLAY_SIZE,
    "sync_target_network": SYNC_TARGET_NETWORK,
    "eps_start": EPS_START,
    "eps_decay": EPS_DECAY
}        
wandb.config.update(env_config)

# Train the agent
agent.train(
    gamma=GAMMA, 
    max_frames=MAX_FRAMES,
    dnn_update_frequency=DNN_UPD, 
    dnn_sync_frequency=DNN_SYNC
)
