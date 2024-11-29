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


import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

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
    images[0].save(save_gif, save_all=True, append_images=images[1:], duration=120, loop=0)      
    return total_reward         
               


MEAN_REWARD_BOUND = 18.0 
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
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

warnings.filterwarnings('ignore')
ENV_NAME = "ALE/Tennis-v5"


env_config = {"skip": 4,
                "stack_size": 4,
                "reshape_size": (84, 84)}

# configuration file

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder


from datetime import datetime 
from stable_baselines3 import DQN, A2C, PPO, SAC
from stable_baselines3.ppo.policies import MlpPolicy
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy


models = ["dqn", "a2c", "ppo"][:1]

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 10,
    "env_name": "CartPole-v1",
    "export_path": "./exports/",
}

def make_env_sb3(render_mode=None):
    env = make_env(ENV_NAME, render_mode=render_mode)
    env = Monitor(env, allow_early_resets=True)
    return env

def eval_model(env, model_name):
    print("Loading and evaluating model '{}'...".format(model_name))

    # load model
    if model_name == "dqn":
        model = DQN.load(config["export_path"] + model_name)
    elif model_name == "a2c":
        model = A2C.load(config["export_path"] + model_name)
    elif model_name == "ppo":
        model = PPO.load(config["export_path"] + model_name)
    elif model_name == "sac":
        model = SAC.load(config["export_path"] + model_name)
    else:
        print("Error, unknown model ({})".format(model_name))
        
    # evaluate the agentç
    obs, info = env.reset()
    mean_reward = []
    for _ in range(100):
        reward_episode = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            reward_episode += reward
            
        mean_reward.append(reward_episode)
        
    mean_reward = np.mean(mean_reward)
    std_reward = np.std(mean_reward)
    
    print("Reward (mean +- std): {:.2f} +- {:.4f}".format(mean_reward, std_reward))

def train_model(env, model_name):
    # Wandb setup
    run = wandb.init(
        project="Tennis",
        config=config,
        name = model_name,      # set run name to model name
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,         # optional
    )

    print("\n>>> Creating and traininig model '{}'...".format(model_name))
    
    # create
    if model_name == "dqn":
        model = DQN(config["policy_type"], env, verbose=0, tensorboard_log=f"runs/{run.id}")
    elif model_name == "a2c":
        model = A2C(config["policy_type"], env, verbose=0, tensorboard_log=f"runs/{run.id}")
    elif model_name == "ppo":
        model = PPO(config["policy_type"], env, verbose=0, tensorboard_log=f"runs/{run.id}")
    elif model_name == "sac":
        model = SAC(config["policy_type"], env, verbose=0, tensorboard_log=f"runs/{run.id}")
    else:
        print("Error, unknown model ({})".format(model_name))
        return None

    # train
    t0 = datetime.now() 
    model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback(verbose=2))
    t1 = datetime.now()
    print('>>> Training time (hh:mm:ss.ms): {}'.format(t1-t0))

    # save and export model
    model.save(config['export_path'] + model_name)
    print("Model exported at '{}'".format(config['export_path'] + model_name))

    # wandb
    run.finish()


env = DummyVecEnv([make_env_sb3])

# train models
#env = make_env_sb3()
#for model_name in models:
#    train_model(env, model_name)
# evaluate models
for model_name in models:
    eval_model(env, model_name)
    
