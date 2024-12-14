import torch
from copy import deepcopy
import numpy as np
import wandb
import warnings
from torch import nn
import torch.optim as optim
import sys

sys.path.append("/ghome/mpilligua/RL/Project_RL/freeway/")
sys.path.append("/ghome/mpilligua/RL/Project_RL/Tennis/")

from DQN_extensions.types_of_buffer import ExperienceReplay, PrioritizedExperienceReplayBuffer
from DQN_extensions.dqn_with_noise import Noisy_DQN

from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import gymnasium as gym

import collections

from PIL import Image
from tqdm import tqdm

import ale_py
from PIL import ImageDraw, ImageFont
import logging
import matplotlib.pyplot as plt



import seaborn

gym.register_envs(ale_py)

q_values_names = {0:"Noop",
                    1:"Fire",
                    2:"Right",
                    3:"Left"}



def get_schedulers(config, optimizer):
    
    
    if config['type_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_max'], eta_min=config['eta_min'])
        
    elif config['type_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
        
    elif config['type_scheduler'] == 'multi':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
    else:
        raise ValueError("Scheduler not recognized")
    return scheduler


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
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, old_shape[1], old_shape[2]), dtype=np.float32)
                                                
    def observation(self, observation):
        # print(observation.shape)
        return observation[:1]

class PongActionWrapper(gym.ActionWrapper):
    def action(self, act):
        # Map [0, 1, 2] to [0, 2, 3]
        mapping = {0: 0, 1: 2, 2: 3}
        return mapping[act]


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)

class CropObservation(gym.ObservationWrapper):
    def __init__(self, env, top, left, height, width):
        super().__init__(env)
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(env.observation_space.shape[0], height, width), dtype=np.uint8)

    def observation(self, obs):
        return obs[:, self.top:self.top+self.height, self.left:self.left+self.width]

def make_env(env_name, skip=4, stack_size=4, reshape_size=(84, 84), only3actions = True, render_mode=None, eval=False):
    env = gym.make(env_name, render_mode=render_mode)
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
    env = Keep_red_dim(env)
    log("Keep_red_dim         : {}".format(env.observation_space.shape))
    env = ReshapeObservation(env, reshape_size)
    log("ReshapeObservation   : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=stack_size)
    log("FrameStackObservation: {}".format(env.observation_space.shape))
    # env = CropObservation(env, 5, 6, 80, 80)    
    # log("CropObservation      : {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    log("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    # env = ForceDifferentAction(env)
    # log("ForceDifferentAction : {}".format(env.observation_space.shape))
    # env = RewardLongPoints(env, return_added_reward=eval)
    # log("RewardLongPoints     : {}".format(env.observation_space.shape))
    
    env = ClipRewardEnv(env)
    
    if only3actions:
        env = PongActionWrapper(env)
        log("PongActionWrapper    : {}".format(env.observation_space.shape))
    env.spec.reward_threshold = 21
    return env

class DQN_NET(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device="cpu", dueling_layer=False):
        super(DQN_NET, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, output_shape)
        self.relu = nn.ReLU()
        self.dueling = dueling_layer

        if dueling_layer:
            self.dueling_layer = nn.Linear(512, output_shape)
        self.to(device)

    def forward(self, x):
        x = self.conv1(x.to(self.device))
        x = self.conv2(self.relu(x))
        x = self.conv3(self.relu(x))
        x = self.conv4(self.relu(x))
        x = self.avgpool(self.relu(x))
        x = x.view(x.size(0), -1)
        x = self.linear1(self.relu(x))
        x = self.linear2(self.relu(x))
        if self.dueling:
            advantage = self.dueling_layer(self.relu(x))
            x += (advantage - advantage.mean())

        return x



class DQN2(torch.nn.Module):# CREDITS: Jordi     
    def __init__(self, input_shape, output_shape, device="cpu", dueling_layer=False):
        super(DQN2, self).__init__()
        self.device = device

        self.model = DQN_NET(input_shape, output_shape, device, dueling_layer)

    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:

            action = np.random.choice(self.actions)
        else:

            qvals = self.get_qvals(state)
            action= torch.max(qvals, dim=-1)[1].item()

        return action

    def forward(self, x):
        return self.model(x)
    
    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])

        state_t = torch.tensor(state, dtype=torch.float, device=self.device)

        return self.model(state_t)
    
    def forward(self, x):
        return self.model(x)


from PIL import Image, ImageDraw, ImageFont

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
            dnnetwork = DQN2(input_shape = self.train_env.observation_space.shape, 
                             output_shape=self.train_env.action_space.n, device=device,
                             dueling_layer=self.dqn_extensions['use_dueling_dqn']).to(device)
        
        # Networks
        self.dnnetwork = dnnetwork.to(device)
        self.target_network = deepcopy(dnnetwork).to(device)
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        self.batch_size = config['batch_size']
        
        # Optimizer for training the network
        self.optimizer = optim.Adam(self.dnnetwork.parameters(), lr=config['learning_rate'])
        
        # Flags to determine which DQN extensions to use
        self.double_dqn = self.dqn_extensions['use_double_dqn']
        self.two_step_dqn = self.dqn_extensions['use_two_step_dqn']
        self.dueling_dqn = self.dqn_extensions['use_dueling_dqn']
        self.use_noise_dqn = self.dqn_extensions['use_noisy_dqn']
        
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
        self.noise_upd_freq = config["noise_upd"]
        
        # Frame counter to keep track of the number of steps taken
        self.frame_number = 0
        
        # Check if the replay buffer uses prioritized experience replay
        self.prioritized_buffer = self.dqn_extensions['use_prioritized_buffer']
        self.results_dir = results_dir
        
        # Reset any noise in the network if using Noisy DQN
        if self.use_noise_dqn:
            self.reset_noise()
        
        # Initialize variables specific to each episode
        self._reset()
        
        
        self.scheduler = get_schedulers(config['scheduler'], self.optimizer) if config['scheduler'] is not None else None
        
        # create_and_save_gif(self.dnnetwork,
                            # eval_env,
                            # device=device, 
                            # save_gif=f"{self.results_dir}/videos/episode_{str(0)}.gif",
                            # epoch=0)
    
        self.episode = 0
        #self.visualize_train(policy='train')
        # exit(0)

    def visualize_train(self, policy='train'):
        if policy == 'train':
            epsilon = self.epsilon
            save_path = f"{self.results_dir}/videos/episode_{self.episode}_train.gif"
        else:
            epsilon = 0
            save_path = f"{self.results_dir}/videos/episode_{self.episode}_eval.gif"

        current_state = self.eval_env.reset()[0]
    
        done = False
        images = []
        total_reward = 0.0
        t = 0
        keep = 0
        const_text_height = 12
        colors = seaborn.color_palette("coolwarm", as_cmap=True)

        max_q_value = 1
        min_q_value = -1

        while not done and t < 5000:
            img = self.eval_env.render()
            # img = img[4:, 8:]
            img_pil = Image.fromarray(img)

            action, q_values = self.get_action(current_state, mode='train', epsilon=epsilon, return_q_values=True)
            current_state, reward, terminated, truncated, _ = self.eval_env.step(action)
            
            img_array = current_state[0]
            img_array = np.expand_dims(current_state[0], 0)
            img_array = np.repeat(img_array, 3, axis=0)
            img_array = np.moveaxis(img_array, 0, -1)  # Move the channel axis to the end
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
            q_values_width = (combined_height-const_text_height) // self.train_env.action_space.n	

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

            for val in range(q_values.shape[1]):
                normalized_q_values = (q_values[0, val] - min_q_value) / (max_q_value - min_q_value)
                color = tuple(int(255 * c) for c in colors(normalized_q_values))
                box = Image.new("RGB", (q_values_width, q_values_width), color=color)
                combined_image.paste(box, (imgs_width, const_text_height+(q_values_width*val)))
                # add text in the middle with the q-value
                q_value_name = q_values_names[val]
                draw.text((imgs_width , const_text_height + (q_values_width*val)), str(val), font=font, fill=(0, 0, 0))

            # Add the epoch number centered at the top of the image
            epoch_text = f"Episode: {self.episode}"
            text_bbox = draw.textbbox((0, 0), epoch_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width) // 2, 1), epoch_text, fill="white", font=font)
        
            # Add the epsilon between the epoch and the reward
            epsilon_text = f"Eps: {epsilon:.2f}"
            text_bbox = draw.textbbox((0, 0), epsilon_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((((combined_image.width - text_width) // 4)*3, 1), epsilon_text, fill="white", font=font)

            # Add the current reward to the top-right corner of the image
            total_reward += reward
            reward_text = f"Reward: {total_reward:.2f}"
            text_bbox = draw.textbbox((0, 0), reward_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((combined_image.width - text_width - 10, 1), reward_text, fill="white", font=font)
            
            # Add the added reward in the top-right corner of the image below the reward
            # if added_reward != 0 or keep:
            #     if keep == 0:
            #         keep = 6
            #         added_reward_text = f"+{added_reward:.2f}"
            #         prev_added_reward = added_reward
            #     else:
            #         added_reward_text = f"+{prev_added_reward:.2f}"
                
            #     text_bbox = draw.textbbox((0, 0), added_reward_text, font=font)
            #     text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            #     draw.text((combined_image.width - text_width - 10, 17), added_reward_text, fill="white", font=font)
            #     keep -= 1

            # Add the action taken to the top-left corner of the image below the steps
            action_text = f"Act: {action}"
            text_bbox = draw.textbbox((0, 0), action_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width) // 4, 1), action_text, fill="white", font=font)

            done = terminated or truncated    
            total_reward += reward  

            images.append(combined_image)

            t += 1
            # print(done, t)

        # save the image
        images[0].save(save_path, save_all=True, append_images=images[1:], duration=100, loop=0)

        self.eval_env.close()
        return total_reward


    def get_action(self, state, mode='explore', epsilon=None, return_q_values=False):
        epsilon = self.epsilon if epsilon is None else epsilon
        q_values = np.zeros((1, self.train_env.action_space.n))

        if mode == 'explore':
            return self.train_env.action_space.sample()
        
        elif mode == "train":
            if np.random.random() < epsilon:
                state_ = np.array([state])
                state_tensor = torch.tensor(state_).to(device)
                q_vals = self.dnnetwork(state_tensor)
                action = self.train_env.action_space.sample()
        
                if return_q_values:
                    return action, q_vals.cpu().detach().numpy()
                return action
            
            else:
                state_ = np.array([state])
                state_tensor = torch.tensor(state_).to(device)
                q_vals = self.dnnetwork(state_tensor)
                _, act_ = torch.max(q_vals, dim=1)
                if return_q_values:
                    q_values = q_vals.cpu().detach().numpy()
                    return int(act_.item()), q_values
                return int(act_.item())
    
    # STEP FROM 2-STEP DQN
    def take_step(self, mode='explore'):
        done_reward = None
        done_number_steps = None
        
        # First Step
        action = self.get_action(self.current_state, mode=mode)
        new_state, reward, terminated, truncated, _ = self.train_env.step(action)

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
            next_action = self.get_action(new_state, mode=mode)
            next_new_state, next_reward, terminated, truncated, _ = self.train_env.step(next_action)
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
        log("Filling replay buffer...")

        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')
            self.frame_number += 1
            if self.use_noise_dqn and (self.frame_number % self.noise_upd_freq == 0):
                self.reset_noise()
            
            print(f"Buffer size: {self.buffer.burn_in_capacity()}", end='\r') 
            
        self.episode = 0
        log("Training...")

        while True:
            self.frame_number +=1
            
            done_reward, done_number_steps = self.take_step(mode='train')

            if self.frame_number % self.dnn_upd_freq == 0:
                    self.update(device)
                    
            if self.frame_number % self.dnn_sync_freq == 0:
                self.target_network.load_state_dict(self.dnnetwork.state_dict())

            if done_reward is not None:
                # If the episode is done, log the results
                self.total_rewards.append(done_reward)
                self.number_of_frames_Episodes.append(done_number_steps)
                mean_reward = np.mean(self.total_rewards[-NUMBER_OF_REWARDS_TO_AVERAGE:])
                
                log(f"Frame:{self.frame_number} | Total games:{len(self.total_rewards)} | Mean reward: {mean_reward:.3f}  (epsilon used: {self.epsilon:.2f})")
                
                wandb.log({"epsilon": self.epsilon, "reward_100": mean_reward, "reward": done_reward,
                            "Frames per episode": done_number_steps}, step=self.frame_number)
        
                self.episode += 1
            
                if self.frame_number >= self.max_frames:
                    log('\nFrame limit reached.')
                    break

                # The game ends if the average reward has reached the threshold
                if mean_reward >= self.train_env.spec.reward_threshold:
                    log('\nEnvironment solved in {} episodes!'.format(self.episode))
                    break
                
                if self.episode % 5 == 0:
                    log("Visualizing the model at episode {}".format(self.episode))
                    self.visualize_train(policy='train')
                    self.visualize_train(policy='eval')

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
        
        if self.scheduler is not None:
           self.scheduler.step()
        
        self.update_loss.append(loss.item())
        wandb.log({"Loss": loss.item()}, step=self.frame_number)
        
    def reset_noise(self):
        for module in self.dnnetwork.modules():
            if hasattr(module, 'reset_noise'):
                module.reset_noise()
    
    def _reset(self):
        self.current_state = self.train_env.reset()[0]
        self.total_reward = 0
        self.number_of_frames_per_episode = 0
        
def epsilon_greedy_policy(Q, state, nA, epsilon):
    '''
    Create a policy where epsilon dictates the probability of a random action being carried out.

    :param Q: linka state -> action value (dictionary)
    :param state: state in which the agent is (int)
    :param nA: number of actions (int)
    :param epsilon: possibility of random movement (float)
    :return: probability of each action (list) d
    '''
    probs = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q[state])
    probs[best_action] += 1.0 - epsilon

    return probs

def setup_buffer(use_prioritized_buffer, config):
    EXPERIENCE_REPLAY_SIZE = config['experience_replay_size']
    config_prioritized_buffer = config['prioritized_buffer_config']
    
    if use_prioritized_buffer:
        buffer = PrioritizedExperienceReplayBuffer(
            memory_size=EXPERIENCE_REPLAY_SIZE,
            alpha=config_prioritized_buffer['alpha'],
            small_constant=config_prioritized_buffer['small_constant'],
            growth_rate=config_prioritized_buffer['growth_rate'],
            beta=config_prioritized_buffer['beta']
        )
    else:
        buffer = ExperienceReplay(capacity=EXPERIENCE_REPLAY_SIZE, burn_in=config_prioritized_buffer['burn_in'])
    return buffer

def log(msg):
    logger.info(msg)
    print(msg, flush=True)

if __name__ == "__main__":
    import os
    import time
    from configs.Rainbow_DQN import * # <--- All hyperparameters are defined here
    import logging
    
    idx = time.strftime("%d%H%M%S")
    name_run = f"Pong_rainbow_{idx}"
    
    run = wandb.init(project="Freeway", name=name_run, entity="pilligua2", group="Pong", mode='disabled')
    results_dir = f"/ghome/mpilligua/RL/Project_RL/Pong/runs/{name_run}"
    #results_dir = f"/fhome/pmlai10/Project_RL/freeway/runs/{name_run}"
    # results_dir =  f"/home/nbiescas/probes/Reinforce/Project_RL/freeway/runs/{name_run}"
    
    
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

    log(f"Initializing train environment with config: {train_env_config}")
    train_env = make_env(**train_env_config)
    
    log(f"\nInitializing eval environment with config: {eval_env_config}")
    eval_env = make_env(**eval_env_config)
    
    # Initialize the 
    
    # MEAN_REWARD_BOUND = 19.0  # self.env.spec.reward... has nothing inside that is why I am using this value
    
    if train_env_config['only3actions']:    # We are only interested in NOOP, RIGHT, LEFT
        train_env.action_space = gym.spaces.Discrete(3)
        eval_env.action_space = gym.spaces.Discrete(3)

    agent = RainbowDQN_Agent(
        train_env=train_env,
        eval_env=eval_env, 
        config=training_config,
        device=device,
        results_dir=results_dir
    )

    wandb.config.update(all_configs)

    log("Training the agent...")
    agent.train()
#