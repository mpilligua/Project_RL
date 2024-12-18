
import torch
import gymnasium as gym
import numpy as np
import time
from PIL import Image
from dataclasses import dataclass, field
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import collections
import ale_py

gym.register_envs(ale_py)

@dataclass
class Agent:
    """
    A class used to store the agent's parameters and memory. 
    Useful for plotting and debugging
    """
    mean_loss: list[int] = field(default_factory=list)
    update_loss: list[int] = field(default_factory=list)
    training_rewards: list[int] = field(default_factory=list)
    mean_training_rewards: list[int] = field(default_factory=list)
    sync_eps: list[int] = field(default_factory=list)
    total_reward: int = 0
    step_count: int = 0
    state0: np.array = None
    

def create_and_save_gif_reinforce(dqn_agent, device, save_gif = 'video.gif'):
  # params
  env = dqn_agent.eval_env
  net = dqn_agent.dnnetwork.to(device)

  # params
  visualize = True

  images = []

  state, _ = env.reset()
  total_reward = 0.0

  while True:
      start_ts = time.time()
      if visualize:
          img = env.render()
          images.append(Image.fromarray(img))

      state_ = torch.tensor([state], device=device)
      q_vals = net(state_)
      action = q_vals.argmax().item()
      state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      total_reward += reward
      if done:
          break


  print("Total reward: %.2f" % total_reward)
  # params
  # duration is the number of milliseconds between frames; this is 40 frames per second
  images[0].save(save_gif, save_all=True, append_images=images[1:], duration=100, loop=0)

  print("Episode export to '{}'".format(save_gif))
  
import torch.nn as nn
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

class Reinforce_net(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Reinforce_net, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
  
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, output_shape)

    def forward(self, obs):

        o = self.conv1(obs)
        o = torch.nn.functional.relu(o)
        o = self.conv2(o)
        o = torch.nn.functional.relu(o)
        o = self.conv3(o)
        o = torch.nn.functional.relu(o)
        
        if o.dim() == 4:
            o = o.view(o.size(0), -1)
        else:
            o = o.view(1, -1)
            
        o = self.fc1(o)
        o = torch.nn.functional.relu(o)
        o = self.fc2(o)
        
        # Apply softmax across action logits
        probs = torch.nn.functional.softmax(o, dim=-1)
        return probs


class Reinforce(Agent):
    def __init__(self, train_env, eval_env, dnnetwork, config, results_dir, device):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env
        self.dnnetwork = dnnetwork
        self.nblock = 100

        self.results_dir = results_dir
        
        self.reward_threshold = MEAN_REWARD_BOUND

        self.training_loss = []
        self.mean_training_loss = []
        self.score = []

        self.optimizer = torch.optim.Adam(self.dnnetwork.parameters(), lr=1e-4)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         config['steps_scheduler'], 
                                                         gamma=config['gamma'], 
                                                         last_epoch=-1)
        self.device = device

    def train(self, GAMMA, MAX_TRAJECTORIES, HORIZON):
        frames = 0
        for trajectory in range(MAX_TRAJECTORIES):
            current_state, _ = self.train_env.reset()
            done = False
            transitions = []

            # Run an entire episode (with maximum number of steps)
            frames_per_episode = 0
            for t in range(HORIZON):
                with torch.no_grad():

                    current_state = torch.from_numpy(current_state).float().to(self.device)
                    actions_prob = self.dnnetwork(current_state)
                    actions_prob.squeeze_()
    
                action = np.random.choice(np.arange(self.train_env.action_space.n), p=actions_prob.cpu().data.numpy())
                previous_state = current_state.cpu()

                current_state, reward, terminated, truncated, info = self.train_env.step(action)

                done = terminated or truncated
                transitions.append((previous_state, action, reward))
                self.score.append(reward)
                frames += 1
                frames_per_episode += 1
                if done:
                    break
                
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

            predicted_batch = self.dnnetwork(state_batch)
            prob_batch = predicted_batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()

            loss = - torch.mean(torch.log(prob_batch) * expected_returns_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.scheduler.step()


            self.training_loss.append(loss.item())
            #wandb.log({"loss": loss.item()})
            self.mean_training_loss.append(np.mean(self.training_loss[-self.nblock:]))

            mean_rewards = np.mean(self.training_rewards[-self.nblock:])
            self.mean_training_rewards.append(mean_rewards)

            wandb.log({"reward": reward_batch_, 
                       "reward_100": mean_rewards, 
                       "Frames per episode": frames_per_episode}, step=frames)

            print('Trajectory {}\tAverage Score: {:.2f}'.format(trajectory, mean_rewards))
            if mean_rewards >= self.reward_threshold:
                print('Environment solved in {} episodes!'.format(trajectory))
                break

            if trajectory % 20 == 0:
                print("Evaluating")
                create_and_save_gif_reinforce(self, self.device, save_gif = f"{self.results_dir}/videos/reinforce_{trajectory}.gif")


def log(msg):
    logger.info(msg)
    print(msg, flush=True)


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
        self.last_actions = collections.deque(maxlen=10)

    def step(self, action):
        self.last_actions.append(action)
        if len(self.last_actions) == 10 and all(a == self.last_actions[0] for a in self.last_actions):
            state, reward, terminated, truncated, info = self.env.step(action)
            return state, reward-100, terminated, truncated, info
       
        
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


if __name__ == "__main__":
    import os
    import time
    import wandb
    from configs.reinforce import * # <--- All hyperparameters are defined here
    import logging
    
    idx = time.strftime("%d%H%M")
    name_run = f"freeway_reinforce_{idx}"
    
    run = wandb.init(project="Freeway", name=name_run, entity="pilligua2")
    #results_dir = f"/ghome/mpilligua/RL/Project_RL/freeway/runs/{name_run}"
    #results_dir = f"/fhome/pmlai10/Project_RL/freeway/runs/{name_run}"
    results_dir =  f"/home/nbiescas/probes/Reinforce/Project_RL/freeway/runs/{name_run}"
    print("Saving to", results_dir)
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


    log(f"Initializing train environment with config: {train_env_config}")
    train_env = make_env(**train_env_config)
    log(f"\nInitializing eval environment with config: {eval_env_config}")
    eval_env = make_env(**eval_env_config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the
    print("Observation,",train_env.observation_space.shape, train_env.action_space.n)
    model = Reinforce_net(train_env.observation_space.shape, train_env.action_space.n).to(device)
  
    MEAN_REWARD_BOUND = 21.0  # self.env.spec.reward... has nothing inside that is why I am using this value
    agent = Reinforce(train_env = train_env, 
                      eval_env = eval_env,
                      device=device, 
                      dnnetwork = model, 
                      config=training_config,
                      results_dir=results_dir)

    
    agent.train(training_config['gamma'], training_config['max_frames'], HORIZON=1000)

    wandb.config.update(all_configs)

    log("Training the agent...")

