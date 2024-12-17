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
from PIL import Image, ImageDraw, ImageFont
import seaborn

gym.register_envs(ale_py)

q_values_names = {0:"Noop",
                    1:"Fire",
                    2:"Right",
                    3:"Left", 
                    4:"RightFire",
                    5:"LeftFire"}


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
            dnnetwork = Noisy_DQN((4, 84, 84), self.train_env.action_space.n, noisy_std = 0.1, device=device).to(device)
        else:
            dnnetwork = DQN2(input_shape = (4, 84, 84), 
                             output_shape=6, device=device,
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
        self.device = device
        
        # self.scheduler = get_schedulers(config['scheduler'], self.optimizer) if config['scheduler'] is not None else None
        
        # create_and_save_gif(self.dnnetwork,
                            # eval_env,
                            # device=device, 
                            # save_gif=f"{self.results_dir}/videos/episode_{str(0)}.gif",
                            # epoch=0)
        self.optimizer = optim.Adam(self.dnnetwork.parameters(), lr=config['learning_rate'])
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

        self.eval_env.reset()
        
        for i in range(113):
            act = 1
            self.eval_env.step(act)
        
        current_state, _, _, _, _ = self.eval_env.last()
    
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
            self.eval_env.step(action)
            current_state, reward, terminated, truncated, _ = self.eval_env.last()
            
            img_array = current_state[:, :, 0]
            img_array = np.expand_dims(img_array, 0)
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
            q_values_width = (combined_height-const_text_height) // 6

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

            # print(q_values.shape)
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
        images[0].save(save_path, save_all=True, append_images=images[1:], duration=20, loop=0)

        # self.eval_env.close()
        return total_reward

    def get_action(self, state, mode='train', epsilon=None, return_q_values=False):
        epsilon = self.epsilon if epsilon is None else epsilon
        q_values = np.zeros((1, 6))

        if mode == 'explore':   
            return np.random.choice(list(range(6)))
        
        elif mode == "train":
            if np.random.random() < epsilon:
                state_ = np.array([state])
                state_tensor = torch.tensor(state_).to(self.device)
                state_tensor = state_tensor.transpose(1, -1)
                q_vals = self.dnnetwork(state_tensor)
                action = np.random.choice(list(range(6)))
                if return_q_values:
                    return action, q_vals.cpu().detach().numpy()
                return action
            
            else:
                state_ = np.array([state])
                state_tensor = torch.tensor(state_).to(self.device)
                state_tensor = state_tensor.transpose(1, -1)
                q_vals = self.dnnetwork(state_tensor)
                _, act_ = torch.max(q_vals, dim=1)
                if return_q_values:
                    q_values = q_vals.cpu().detach().numpy()
                    return int(act_.item()), q_values
                return int(act_.item())
    
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
        
    def calculate_loss(self, batch):
        # states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
        states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
        # print([b[1] for b in batch])
        # states = np.array([b[0] for b in batch])
        # actions = np.array([b[1] for b in batch])
        # rewards = np.array([b[2] for b in batch])
        # dones = np.array([b[3] for b in batch])
        # next_states = np.array([b[4] for b in batch])
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.LongTensor(actions).view(-1, 1).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # go from  (84, 84, 4, batch_size) to (batch_size, 4, 84, 84)
        states = torch.tensor(states).to(self.device)
        states = states.permute(0, 3, 1, 2)
        
        next_states = torch.tensor(next_states).to(self.device)
        next_states = next_states.permute(0, 3, 1, 2)
        
        qvals = self.dnnetwork(states).gather(1, actions)
        qvals_next = torch.max(self.target_network(next_states), dim=1)[0].detach()
                
        qvals_next[dones] = 0
        
        expected_qvals = rewards + self.gamma * qvals_next

        # Loss for the prioritized buffer
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
        # self.current_state = self.train_env.reset()[0]
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

class loggerr():
    def __init__(self, logger):
        self.logger = logger
    def log(self, msg):
        self.logger.info(msg)
        print(msg, flush=True)
