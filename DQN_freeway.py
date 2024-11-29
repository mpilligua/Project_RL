import gymnasium as gym
import warnings
import numpy as np
import gymnasium
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import torch
import torch.nn as nn        
import torch.optim as optim 
import collections
from PIL import Image

import wandb

import ale_py

gym.register_envs(ale_py)

class ImageToPyTorch(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name, skip = 4, stack_size=4, reshape_size = (84, 84), render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    print("Standard Env.        : {}".format(env.observation_space.shape))
    env = MaxAndSkipObservation(env, skip=skip)
    print("MaxAndSkipObservation: {}".format(env.observation_space.shape))
    #env = FireResetEnv(env)
    env = ResizeObservation(env, reshape_size)
    print("ResizeObservation    : {}".format(env.observation_space.shape))
    env = GrayscaleObservation(env, keep_dim=True)
    print("GrayscaleObservation : {}".format(env.observation_space.shape))
    env = ImageToPyTorch(env)
    print("ImageToPyTorch       : {}".format(env.observation_space.shape))
    env = ReshapeObservation(env, (reshape_size))
    print("ReshapeObservation   : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=stack_size)
    print("FrameStackObservation: {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    
    return env



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

warnings.filterwarnings('ignore')
ENV_NAME = "ALE/Freeway-v5"
    
    
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

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, BATCH_SIZE):
        indices = np.random.choice(len(self.buffer), BATCH_SIZE, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)
          
def create_and_save_gif(net, save_gif = 'video.gif'): # CREDITS JORDI
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
               
               
class Agent:
    def __init__(self, env, exp_replay_buffer):
        self.env = env
        self.exp_replay_buffer = exp_replay_buffer
        self._reset()

    def _reset(self):
        self.current_state = self.env.reset()[0]
        self.total_reward = 0.0

    def step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_ = np.array([self.current_state])
            state = torch.tensor(state_).to(device)
            q_vals = net(state)
            _, act_ = torch.max(q_vals, dim=1)
            action = int(act_.item())

        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward

        exp = Experience(self.current_state, action, reward, is_done, new_state)
        self.exp_replay_buffer.append(exp)
        self.current_state = new_state
        
        if is_done:
            done_reward = self.total_reward
            self._reset()
        
        return done_reward


MEAN_REWARD_BOUND = 19.0 
NUMBER_OF_REWARDS_TO_AVERAGE = 10          

GAMMA = 0.99   
    
BATCH_SIZE = 32  
LEARNING_RATE = 1e-4           

EXPERIENCE_REPLAY_SIZE = 10000            
SYNC_TARGET_NETWORK = 1000     

EPS_START = 1.0
EPS_DECAY = 0.999985
EPS_MIN = 0.02


# start a new wandb run to track this script
import os
idx = os.listdir("/home/nbiescas/Desktop/freeway").__len__()
wandb.init(project="Freeway", name=f"freeway_{idx}")
os.makedirs(f"/home/nbiescas/Desktop/freeway/freeway_{idx}", exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

warnings.filterwarnings('ignore')
ENV_NAME = "ALE/Freeway-v5"


env_config = {"skip": 4,
                "stack_size": 4,
                "reshape_size": (84, 84),
                "render_mode": None}

env = make_env(ENV_NAME)

wandb.config.update(env_config)

print(env.observation_space.shape, env.action_space.n)
net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
target_net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
 
buffer = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
agent = Agent(env, buffer)

epsilon = EPS_START
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
total_rewards = []
frame_number = 0  

while True:
    frame_number += 1
    epsilon = max(epsilon * EPS_DECAY, EPS_MIN)

    reward = agent.step(net, epsilon, device=device)
    if frame_number % 10000 == 0:
        create_and_save_gif(net.to("cuda"), save_gif=f"/home/nbiescas/Desktop/freeway/freeway_{idx}/frame_" + str(frame_number) + ".gif")
        
    if reward is not None:
        total_rewards.append(reward)

        mean_reward = np.mean(total_rewards[-NUMBER_OF_REWARDS_TO_AVERAGE:])
        print(f"Frame:{frame_number} | Total games:{len(total_rewards)} | Mean reward: {mean_reward:.3f}  (epsilon used: {epsilon:.2f})")
        wandb.log({"epsilon": epsilon, "reward_100": mean_reward, "reward": reward}, step=frame_number)

        if mean_reward > MEAN_REWARD_BOUND:
            print(f"SOLVED in {frame_number} frames and {len(total_rewards)} games")
            break

    if len(buffer) < EXPERIENCE_REPLAY_SIZE:
        continue

    batch = buffer.sample(BATCH_SIZE)
    states_, actions_, rewards_, dones_, next_states_ = batch

    states = torch.tensor(states_).to(device)
    next_states = torch.tensor(next_states_).to(device)
    actions = torch.tensor(actions_).to(device)
    rewards = torch.tensor(rewards_).to(device)
    dones = torch.BoolTensor(dones_).to(device)

    Q_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    next_state_values = target_net(next_states).max(1)[0]
    next_state_values[dones] = 0.0
    next_state_values = next_state_values.detach()

    expected_Q_values = next_state_values * GAMMA + rewards
    loss = nn.MSELoss()(Q_values, expected_Q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if frame_number % SYNC_TARGET_NETWORK == 0:
        target_net.load_state_dict(net.state_dict())