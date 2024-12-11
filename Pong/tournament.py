import os
import time
from configs.Rainbow_DQN import * # <--- All hyperparameters are defined here
import logging
import supersuit as ss
from pettingzoo.atari import pong_v3
import numpy as np
import wandb
from tournament_utils import *

def make_env():
    # Load the environment
    env = pong_v3.env(render_mode="human")

    # Pre-process using SuperSuit
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.resize_v1(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 4)
    # env = ss.dtype_v0(env, dtype=np.float32)
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # obs_space = env.observation_space(env.possible_agents[1])
    # env.observation_space = env.observation_space(env.possible_agents[1])
    # env = ss.transpose_v0(env, (2, 0, 1))
    
    return env

# idx = time.strftime("%d%H%M%S")
# name_run = f"Pong_rainbow_{idx}"

# run = wandb.init(project="Freeway", name=name_run, entity="pilligua2", group="Pong", mode='disabled')
# results_dir = f"/ghome/mpilligua/RL/Project_RL/Pong/runs/{name_run}"

# os.makedirs(results_dir, exist_ok=True)
# os.makedirs(f"{results_dir}/videos", exist_ok=True)
# # initialize logger.logging file

# logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         filename=f"{results_dir}/log.log",
#     )
# ini_logger = logging.getLogger(__name__)
# logger = loggerr(ini_logger)

# logger.log(msg=f"Saving to: {results_dir}")
# logger.log(f"Wandb run: {name_run} - {run.id}")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_env = make_env()
# eval_env = make_env()

# # Load the agents
# model1 = RainbowDQN_Agent(train_env=train_env,
#                             eval_env=eval_env, 
#                             config=training_config,
#                             device=device,
#                             results_dir=results_dir)

# model2 = RainbowDQN_Agent(train_env=train_env,
#                             eval_env=eval_env, 
#                             config=training_config,
#                             device=device,
#                             results_dir=results_dir)

# rewards = {agent: 0 for agent in train_env.possible_agents}


# # We evaluate here using an AEC environments
# train_env.reset(seed=1234)

# train_env.action_space(train_env.possible_agents[0]).seed(0)
# current_state, _, _, _,  _ = train_env.last() 

# # init with reset

# history = []

# dnn_upd_freq = 10
# dnn_sync_freq = 1000
# total_frames = 10

# frame_number = 0
# all_episodes_rewards = {0: [], 1: []}
# num_frames_episodes = {0: [], 1: []}
# mean_reward = {0: 0, 1: 0}
# episodes = 0
# batch_size = 2
# for _ in range(total_frames):

#     total_rewards = {0: 0, 1: 0}
#     number_of_frames_per_episode = {0: 0, 1: 0}

#     for agent in train_env.agent_iter():
#         if agent == train_env.possible_agents[0]:
#             act = model1.get_action(current_state)
#         else:
#             act = model2.get_action(current_state)

#         train_env.step(act)
#         next_state, reward, termination, truncation, info = train_env.last()
        
#         done = termination or truncation
#         history.append([current_state, act, reward, done, next_state])
#         current_state = next_state

#         if len(history) < batch_size:
#             continue

#         if agent == train_env.possible_agents[0]:
#             total_rewards[0] += reward
#         else:
#             total_rewards[1] += reward

#         if done:
#             train_env.reset()
#             current_state, _, _, _, _ = train_env.last()
#             total_rewards = {0: 0, 1: 0}
#             number_of_frames_per_episode = {0: 0, 1: 0}

#         if (frame_number % dnn_upd_freq == 0) or (frame_number % (dnn_upd_freq+1) == 0):
#             batch = history[-batch_size:]

#             if agent == train_env.possible_agents[0]:
#                 loss = model1.calculate_loss(batch)
#             else:
#                 loss = model2.calculate_loss(batch)

#             loss.backward()
#             if agent == train_env.possible_agents[0]:
#                 model1.optimizer.step()
#             else:
#                 model2.optimizer.step()

#         if (frame_number % dnn_sync_freq == 0):
#             model1.target_network.load_state_dict(model1.dnnetwork.state_dict())
#             model2.target_network.load_state_dict(model2.dnnetwork.state_dict())

#         if done:
#             all_episodes_rewards[0].append(total_rewards[0])
#             all_episodes_rewards[1].append(total_rewards[1])
            
#             num_frames_episodes[0].append(number_of_frames_per_episode[0])
#             num_frames_episodes[1].append(number_of_frames_per_episode[1])

#             mean_reward[0] = np.mean(all_episodes_rewards[0][-100:])
#             mean_reward[1] = np.mean(all_episodes_rewards[1][-100:])

#             logger.log(f"Frame:{frame_number} | Games:{episodes} | Mean reward A1: {mean_reward[0]:.2f} | Mean reward A2: {mean_reward[1]:.2f}")

#             episodes += 1
#             model1.episode = episodes
#             model2.episode = episodes

#             # if episodes % 1 == 0:
#             logger.log(f"Visualizing the model at episode {episodes}")
#             model1.visualize_train(policy='train')
#             model1.visualize_train(policy='eval')

#             model2.visualize_train(policy='train')
#             model2.visualize_train(policy='eval')

#         frame_number += 1

#     train_env.close()


env = make_env()

env.reset()
for i in range(20):
    action = np.random.choice(5)
    env.step(action)
    current_state, _, _, _, _ = env.last()
    print(action)

    # plt.imshow(current_state[:, :, 0])
    name = f"test_{i}.png"
    plt.imsave(name, current_state[..., 0], cmap='gray')