import os
import time
from configs.Rainbow_DQN import * # <--- All hyperparameters are defined here
import logging
import supersuit as ss
from pettingzoo.atari import pong_v3
import numpy as np
import wandb
from Pong.MAS_tournament_dqn_utils import *

def save_image(render, obs, q_values, action, reward, epsilon, save_path, episode, t, total_reward):
    min_q_value = -1
    max_q_value = 1
    const_text_height = 12
    colors = seaborn.color_palette("coolwarm", as_cmap=True)
    img_pil = Image.fromarray(render)
    
    img_array = obs[:, :, 0]
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
    epoch_text = f"Episode: {episode}"
    text_bbox = draw.textbbox((0, 0), epoch_text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    draw.text(((combined_image.width - text_width) // 2, 1), epoch_text, fill="white", font=font)

    # Add the epsilon between the epoch and the reward
    epsilon_text = f"Eps: {epsilon:.2f}"
    text_bbox = draw.textbbox((0, 0), epsilon_text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    draw.text((((combined_image.width - text_width) // 4)*3, 1), epsilon_text, fill="white", font=font)

    # Add the current reward to the top-right corner of the image
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
 

    return combined_image

    # t += 1
    # # print(done, t)

    # # save the image
    # images[0].save(save_path, save_all=True, append_images=images[1:], duration=20, loop=0)

    # self.eval_env.close()
    # return total_reward

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
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=4, base_class="gym")
    return env

idx = time.strftime("%d%H%M%S")
name_run = f"Pong_rainbow_{idx}"

run = wandb.init(project="Freeway", name=name_run, entity="pilligua2", group="Pong", mode='disabled')
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
eval_env = make_env()

# Load the agents
model1 = RainbowDQN_Agent(train_env=train_env,
                            eval_env=eval_env, 
                            config=training_config,
                            device=device,
                            results_dir=results_dir)

model2 = RainbowDQN_Agent(train_env=train_env,
                            eval_env=eval_env, 
                            config=training_config,
                            device=device,
                            results_dir=results_dir)

rewards = {agent: 0 for agent in train_env.possible_agents}


# We evaluate here using an AEC environments
train_env.reset(seed=1234)

train_env.action_space(train_env.possible_agents[0]).seed(0)
current_state, _, _, _,  _ = train_env.last() 

# SKIP UNNECESSARY FRAMES
for i in range(114):
    act = 1
    train_env.step(act)
    # next_state, reward, termination, truncation, info = train_env.last()
        
# init with reset

history = []

dnn_upd_freq = 10
dnn_sync_freq = 1000
total_frames = 1000000

frame_number = 0
all_episodes_rewards = {0: [], 1: []}
num_frames_episodes = {0: [], 1: []}
mean_reward = {0: 0, 1: 0}
episodes = 0
epsilon = training_config["eps_start"]
batch_size = 2
images = []
# total_reward = 
for _ in range(total_frames):

    total_rewards = {0: 0, 1: 0}
    number_of_frames_per_episode = {0: 0, 1: 0}

    for agent in train_env.agent_iter():
        if agent == train_env.possible_agents[0]:
            act, q_values = model1.get_action(current_state, return_q_values=True)
        else:
            act, q_values = model2.get_action(current_state, return_q_values=True)

        train_env.step(act)
        next_state, reward, termination, truncation, info = train_env.last()
        
        done = termination or truncation
        history.append([current_state, act, reward, done, next_state])
        current_state = next_state

        if episodes % 2 == 0:
            render = train_env.render()
            img = save_image(render, current_state, q_values, act, reward, epsilon, f"{results_dir}/videos/{frame_number}.png", episodes, frame_number, total_rewards[0])
            images.append(img)

        if len(history) < batch_size:
            continue

        if agent == train_env.possible_agents[0]:
            total_rewards[0] += reward
        else:
            total_rewards[1] += reward

        if done:
            train_env.reset()
            current_state, _, _, _, _ = train_env.last()
            total_rewards = {0: 0, 1: 0}
            number_of_frames_per_episode = {0: 0, 1: 0}

        if (frame_number % dnn_upd_freq == 0) or (frame_number % (dnn_upd_freq+1) == 0):
            batch = history[-batch_size:]

            if agent == train_env.possible_agents[0]:
                loss = model1.calculate_loss(batch)
            else:
                loss = model2.calculate_loss(batch)

            loss.backward()
            if agent == train_env.possible_agents[0]:
                model1.optimizer.step()
            else:
                model2.optimizer.step()

        if (frame_number % dnn_sync_freq == 0):
            model1.target_network.load_state_dict(model1.dnnetwork.state_dict())
            model2.target_network.load_state_dict(model2.dnnetwork.state_dict())

        if done:
            all_episodes_rewards[0].append(total_rewards[0])
            all_episodes_rewards[1].append(total_rewards[1])
            
            num_frames_episodes[0].append(number_of_frames_per_episode[0])
            num_frames_episodes[1].append(number_of_frames_per_episode[1])

            mean_reward[0] = np.mean(all_episodes_rewards[0][-100:])
            mean_reward[1] = np.mean(all_episodes_rewards[1][-100:])

            logger.log(f"Frame:{frame_number} | Games:{episodes} | Mean reward A1: {mean_reward[0]:.2f} | Mean reward A2: {mean_reward[1]:.2f}")

            episodes += 1
            model1.episode = episodes
            model2.episode = episodes

            # if episodes % 1 == 0:
            logger.log(f"Visualizing the model at episode {episodes}")
            # model1.visualize_train(policy='train')
            # model1.visualize_train(policy='eval')

            # model2.visualize_train(policy='train')
            # model2.visualize_train(policy='eval')

            if episodes % 2 == 0:
                images[0].save(f"{results_dir}/videos/{episodes}.gif", save_all=True, append_images=images[1:], duration=20, loop=0)

        frame_number += 1
        model1.epsilon = epsilon
        model2.epsilon = epsilon

    epsilon = epsilon * training_config["eps_decay"]
    train_env.close()


# env = make_env()

# env.reset()
# env.action_space(env.possible_agents[0]).seed(0)
# # current_state, _, _, _,  _ = env.last()

# for i in range(200):
#     # for j, agent in enumerate(env.agent_iter()):
#     action = 1
#     print(action)
#     env.step(action)
#     # action = 5
#     # env.step(action)
#     current_state = env.observe(env.possible_agents[0])
#     # print(action)
#     # print(current_state.shape)

#     # plt.imshow(current_state[:, :, 0])
#     name = f"test_{i}.png"
#     plt.imsave(name, current_state)