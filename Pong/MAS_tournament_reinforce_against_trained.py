import os
import time
from configs.Rainbow_DQN import * # <--- All hyperparameters are defined here
import logging
import supersuit as ss
from pettingzoo.atari import pong_v3
import numpy as np
import wandb
from Pong.MAS_tournament_dqn_utils import *
import matplotlib
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SingleAgentEnv(gym.Env):
    """Wraps the multi-agent environment to allow a single agent to interact in a turn-based fashion."""
    def __init__(self, env):
        self.env = env
        print(f"Action space: {self.env.action_space}")
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84, 4), dtype=np.float32)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        observations, _, _, _, infos = self.env.last()
        return observations, infos

    def step(self, action):
        # print(f"Taking action: {action}")
        self.env.step(action)
        obs, rewards, terminated, truncated, infos = self.env.last()
        return (
            obs,
            rewards,
            terminated,
            truncated,
            infos,
        )

    def agent_iter(self):
        for agent in self.env.agent_iter():
            yield agent
            
    def render(self, **kwargs):
        return self.env.render(**kwargs)


def make_env(num_envs=1):
    # Load the environment
    env = pong_v3.env(
                    # frameskip=1,
                    render_mode="rgb_array"
                    # max_episode_steps=max_episode_steps,
                )

    # Pre-process using SuperSuit
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    
    # env = ss.pettingzoo_env_to_vec_env_v0(env)
    # env = ss.concat_vec_envs_v1(env, num_envs, base_class='gymnasium')
    return env



def visualize_train(env, model1, model2, episode =100000, save_path = ""):
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        current_state, _ = env.reset()
        # current_state, _, _, _, _ = train_env.last()
    
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
        for agent in env.agent_iter():
            img = env.render()
            img = img.astype(np.uint8)
            img_pil = Image.fromarray(img)

            with torch.no_grad():
                current_state = torch.from_numpy(current_state).float().to(device)
                if agent == "first_0":
                    # actions_prob = model1.policy_network(current_state.unsqueeze(0))
                    action, _ = model1.predict(current_state.unsqueeze(0).cpu())
                    action = action.item()
                else:
                    actions_prob = model2.policy_network(current_state.unsqueeze(0))
                    actions_prob.squeeze_()
                    action = np.random.choice(np.arange(6), p=actions_prob.cpu().data.numpy())
            
            previous_state = current_state.cpu()
            
            current_state, reward, terminated, truncated, _ = train_env.step(action)
            
            img_array = np.expand_dims(current_state, 0)
            img_array = np.repeat(img_array, 3, axis=0)
            img_array = np.moveaxis(img_array, 0, -1)  # Move the channel axis to the end
            img_array = (img_array * 255).astype(np.uint8)  # Convert to uint8
            img_array = img_array[:, :, -1]
            img_input_model = Image.fromarray(img_array)

            # Ensure both images have the same height
            height = max(img_pil.height, img_input_model.height)
            img_pil = img_pil.resize((img_pil.width, height))
            new_width = int(img_input_model.width * (height / img_input_model.height))
            img_input_model = img_input_model.resize((new_width, height))

            # Calculate the dimensions for the final image
            imgs_width = img_pil.width + img_input_model.width
            combined_height = height + const_text_height
            probabilities_width = (combined_height-const_text_height) // 6

            total_width = imgs_width + probabilities_width*2

            # Create a blank canvas with extra space for text
            combined_image = Image.new("RGB", (total_width, combined_height), color=(0, 0, 0))

            # Paste the images onto the canvas
            combined_image.paste(img_pil, (probabilities_width, const_text_height))
            combined_image.paste(img_input_model, ((probabilities_width*2)+img_pil.width, const_text_height))

            # Add a black bar with the text above the images
            box = Image.new("RGB", (total_width, const_text_height), color=(0, 0, 0))
            combined_image.paste(box, (0, 0))

            draw = ImageDraw.Draw(combined_image)

            # Add the step number to the upper-left corner of the image
            font = ImageFont.load_default() 
            step_text = f"Step: {t}"
            text_bbox = draw.textbbox((0, 0), step_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((2*(combined_image.width - text_width)//3, 1), step_text, fill="white", font=font)
            
            # Add the epoch number centered at the top of the image
            epoch_text = f"Episode: {episode}"
            text_bbox = draw.textbbox((0, 0), epoch_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((combined_image.width - text_width), 1), epoch_text, fill="white", font=font)
        
            if agent == "first_0":
                action_model2 = action
                reward_model1 += reward
                # probabs_model1 = actions_prob
            else:
                action_model1 = action
                reward_model2 += reward
                probabs_model2 = actions_prob
                
        
            # Add the current reward to the top-right corner of the image
            reward_text = f"Reward: {reward_model1:.2f}"
            text_bbox = draw.textbbox((0, 0), reward_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((probabilities_width+5, 10), reward_text, fill="white", font=font)
            
            # Add the action taken to the top-left corner of the image below the steps
            action_text = f"Act: {action_model1}"
            text_bbox = draw.textbbox((0, 0), action_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((probabilities_width+5, 1), action_text, fill="white", font=font)

            # Add the current reward to the top-right corner of the image
            reward_text = f"Reward: {reward_model2:.2f}"
            text_bbox = draw.textbbox((0, 0), reward_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((probabilities_width+img_pil.width - 5-text_width, 10), reward_text, fill="white", font=font)
            
            # Add the action taken to the top-left corner of the image below the steps
            action_text = f"Act: {action_model2}"
            text_bbox = draw.textbbox((0, 0), action_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((probabilities_width+img_pil.width - 5-text_width, 1), action_text, fill="white", font=font)

            for i in range(6):
                probab = probabs_model2[i].item()
                color = tuple((np.array(colors(probab)) * 255).astype(int))
                draw.rectangle([0, const_text_height+i*probabilities_width, probabilities_width, const_text_height+(i+1)*probabilities_width], fill=color)
            
            for i in range(6):
                probab = probabs_model1[i].item()
                color = tuple((np.array(colors(probab)) * 255).astype(int))
                draw.rectangle([probabilities_width+img_pil.width, const_text_height+i*probabilities_width, probabilities_width+img_pil.width+probabilities_width, const_text_height+(i+1)*probabilities_width], fill=color)


            images.append(combined_image)
            
            if terminated or truncated or t > 3000:
                break


            t += 1

        # save the image
        images[0].save(save_path, save_all=True, append_images=images[1:], duration=20, loop=0)

        env.close()
        return total_reward
    
    

class Reinforce_net(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Reinforce_net, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
  
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, output_shape)

    def forward(self, obs):
        # print(obs.shape)
        obs = obs.permute(0, 3, 1, 2)

        o = self.conv1(obs)
        o = torch.nn.functional.relu(o)
        o = self.conv2(o)
        o = torch.nn.functional.relu(o)
        o = self.conv3(o)
        o = torch.nn.functional.relu(o)
        
        if o.dim() == 4:
            o = o.reshape(o.size(0), -1)
        else:
            o = o.view(1, -1)
            
        o = self.fc1(o)
        o = torch.nn.functional.relu(o)
        o = self.fc2(o)
        
        o = torch.clamp(o, min=-1000, max=1000)
        # Apply softmax across action logits
        probs = torch.nn.functional.softmax(o, dim=-1)
        return probs

class ReinforceAgent:
    def __init__(self, env, input_dim, output_dim, lr=1e-3, device="cpu"):
        self.env = env
        self.policy_network = Reinforce_net(input_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []
        self.transitions = []
        self.device = device
        self.training_rewards = []
    
    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs = self.policy_network(state_tensor)
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def update_policy(self):
        transitions = self.transitions
        reward_batch = torch.tensor(np.array([r for (s,a,r) in transitions]), device = self.device).flip(dims=(0,))
        state_batch = torch.tensor(np.array([s for (s,a,r) in transitions]), device = self.device)
        action_batch = torch.tensor(np.array([a for (s,a,r) in transitions]), device = self.device)

        reward_batch_ = reward_batch.sum().item()

        self.training_rewards.append(reward_batch_); #wandb.log("reward", reward_batch_)

        batch_Gvals = []
        R = 0
        for i in range(len(transitions)):
            R = reward_batch[i] + GAMMA * R
            batch_Gvals.append(R)
        batch_Gvals = torch.FloatTensor(batch_Gvals[::-1]).to(self.device)

        expected_returns_batch = (batch_Gvals - batch_Gvals.mean()) / (batch_Gvals.std() + 1e-9)

        predicted_batch = self.policy_network(state_batch)
        prob_batch = predicted_batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()

        loss = - torch.mean(torch.log(prob_batch) * expected_returns_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.scheduler.step()
        return loss.item()
        

idx = time.strftime("%d%H%M%S")
name_run = f"MAS_PongReinforce{idx}"

run = wandb.init(project="Freeway", name=name_run, entity="pilligua2", group="Pong")
results_dir = f"/ghome/mpilligua/RL/Project_RL/Pong/runs/{name_run}"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(f"{results_dir}/videos", exist_ok=True)
# initialize logger.logging file

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename=f"{results_dir}/log.log",
    )
ini_logger = logging.getLogger(__name__)
logger = loggerr(ini_logger)

logger.log(msg=f"Saving to: {results_dir}")
logger.log(f"Wandb run: {name_run} - {run.id}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_env = make_env()
train_env = SingleAgentEnv(train_env)

eval_env = make_env()


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
        observations = observations.permute(0, 3, 1, 2)
        # print(observations.shape)
        return self.cnn(observations)



policy_kwargs = {
    'features_extractor_class': CustomCNN,
    'features_extractor_kwargs': {'features_dim': 3},
    'normalize_images': False
}

# Load the agents
model1 = PPO("CnnPolicy", train_env, verbose=1, tensorboard_log=f"{results_dir}/tensorboard/", policy_kwargs=policy_kwargs, device=device)
model1.load(f"/ghome/mpilligua/RL/Project_RL/Pong/runs/PongPPO5/ppo_pong.zip")
# model1 = ReinforceAgent(train_env, input_dim=(4, 84, 84), output_dim=6, device=device)
model2 = ReinforceAgent(train_env, input_dim=(4, 84, 84), output_dim=6, device=device)

# rewards = {agent: 0 for agent in train_env.possible_agents}


# We evaluate here using an AEC environments
# train_env.reset(seed=1234)

# train_env.action_space(train_env.possible_agents[0]).seed(0)
# current_state, _, _, _,  _ = train_env.last() 

history = []

dnn_upd_freq = 10
dnn_sync_freq = 1000
total_frames = 1000000

all_episodes_rewards = {0: [], 1: []}
num_frames_episodes = {0: [], 1: []}
mean_reward = {0: 0, 1: 0}
episodes = 0
epsilon = training_config["eps_start"]
batch_size = 2
images = []
GAMMA = 0.9
frames = 0
MAX_TRAJECTORIES = 1000
HORIZON = 1000
training_rewards = []
scores = []

for trajectory in range(MAX_TRAJECTORIES):
    current_state, _ = train_env.reset()
    # current_state, _, _, _, _ = train_env.last()
    done = False
    model1.transitions = []
    model2.transitions = []
    frames_per_episode = 0
    for agent in train_env.agent_iter():
        with torch.no_grad():
            current_state = torch.from_numpy(current_state).float().to(device)
            if agent == "first_0":
                # actions_prob = model1.policy_network(current_state.unsqueeze(0))
                action, _ = model1.predict(current_state.unsqueeze(0).cpu())
                action = action.item()
            else:
                actions_prob = model2.policy_network(current_state.unsqueeze(0))
                actions_prob.squeeze_()
                action = np.random.choice(np.arange(6), p=actions_prob.cpu().data.numpy())
        
        previous_state = current_state.cpu()
        
        current_state, reward, terminated, truncated, _ = train_env.step(action)
        # current_state, reward, terminated, truncated, _ = train_env.last()
        
        if agent != "first_0":
            # model1.transitions.append((previous_state, action, reward))
        # else:
            model2.transitions.append((previous_state, action, reward))
        
        frames += 1
        frames_per_episode += 1
        
        if frames_per_episode > 6000:
            break
        
        if terminated or truncated:
            break
    
    print("Updating policies")
    # loss1 = model1.update_policy()
    loss2 = model2.update_policy()
    
    wandb.log({"loss_model2": loss2, "frames": frames, "episode": trajectory})
    
    if trajectory % 1 == 0:
        print(f"Visualizing trajectory {trajectory}")
        save_path = f"{results_dir}/videos/{trajectory}.gif"
        visualize_train(train_env, model1, model2, trajectory, f"{results_dir}/videos/{trajectory}.gif")
        
        if trajectory % 50 == 0:
            wandb.log({"videos": wandb.Video(save_path, fps=4, format="gif")})
            

# Save the models
# torch.save(model1.policy_network.state_dict(), f"{results_dir}/model1.pth")
torch.save(model2.policy_network.state_dict(), f"{results_dir}/model2.pth")

run.finish()
