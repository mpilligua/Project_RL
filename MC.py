import gymnasium as gym
import ale_py
import numpy as np
from collections import defaultdict
import sys
from tqdm import tqdm
import torch
import time


def detect_cars_and_directions(array, cars_top, cars_bottom, index, device="cuda"):
    start_time = time.time()  # Start time for this section
    detected_car = array[cars_top[index]:cars_bottom[index]+1, :, :].to(device)

    # Define the road color
    color_road = torch.tensor([142, 142, 142], device=device)

    # Create a boolean mask for pixels matching the road color
    bbox = torch.all(detected_car == color_road, dim=2)  # Shape: (H, W)

    # Define the pattern to search
    pattern2search = torch.tensor([True, False, False, False, False, False, False, True], device=device)

    # Find columns where the pattern matches
    found = torch.all(bbox == pattern2search.view(-1, 1), dim=0)

    # Slide the found pattern to check for the reverse direction
    slided_found = torch.cat([torch.tensor([False], device=device), found[:-1]])

    # Combine the mask with the road alignment
    all_road = torch.all(bbox, dim=0)
    slided_found = torch.logical_and(all_road, slided_found)
    slided_found = torch.cat([slided_found[1:], torch.tensor([False], device=device)])

    # Determine car direction:
    # - 1 indicates car moving left, 2 indicates car moving right
    cars_and_direction = found.to(torch.int32) + slided_found.to(torch.int32)
    
    end_time = time.time()  # End time for this section
    print(f"detect_cars_and_directions time: {end_time - start_time:.4f} seconds")
    
    return cars_and_direction


def find_bunny(array_with_roads, final_array, roads, device="cuda"):
    start_time = time.time()  # Start time for this section
    yellow = torch.tensor([252, 252, 84], device=device)
    yellow_pixels = torch.all(array_with_roads == yellow, dim=2)
    yellow_pixels_shifted = torch.cat([yellow_pixels[1:], yellow_pixels[:1]], dim=0)
    bunnies = torch.logical_and(yellow_pixels, yellow_pixels_shifted)

    x_bunny = torch.any(bunnies, dim=0)
    y_bunny = torch.any(bunnies, dim=1)

    bunny_road = None
    for i in range(len(roads), -1, -1):
        if i == 0:
            found = torch.any(y_bunny[roads[i]:])
        elif i == len(roads):
            found = torch.any(y_bunny[:roads[i-1] + 1])
        else:
            found = torch.any(y_bunny[roads[i]:roads[i-1] + 1])
        
        if found:
            bunny_road = i
            break

    final_array[bunny_road, x_bunny] = 3
    end_time = time.time()  # End time for this section
    print(f"find_bunny time: {end_time - start_time:.4f} seconds")


def frame2state(final_array):
    start_time = time.time()  # Start time for this section
    final_array_copy = final_array.clone()
    pos_bunny = torch.where(final_array_copy == 3)
    bunny_x = pos_bunny[1][4]
    bunny_y = pos_bunny[0][0]
    
    final_array_copy = final_array_copy[:bunny_y+2, :bunny_x]
    final_array_copy = (final_array_copy == 1) | (final_array_copy == 2)
    
    state = tuple(final_array_copy.flip(0).reshape(-1).tolist())
    end_time = time.time()  # End time for this section
    print(f"frame2state time: {end_time - start_time:.4f} seconds")
    
    return state, bunny_y

def process_frame(array, roads, cars_top, cars_bottom, seen_roads, device):
    start_time = time.time()  # Start time for this section
    final_array = torch.zeros((len(roads)-1, array.shape[1]), dtype=int, device=device)
    array = torch.tensor(array, dtype=int, device=device)
    for road in range(len(roads)-1):
        result = detect_cars_and_directions(array, cars_top, cars_bottom, road)
        final_array[(len(roads)-2)-road] = result

    find_bunny(array, final_array, roads)
    state, current_road = frame2state(final_array)

    reward = 0
    if seen_roads[current_road] == 0:
        seen_roads[current_road] = 1
        reward = 1
    end_time = time.time()  # End time for this section
    print(f"process_frame time: {end_time - start_time:.4f} seconds")
    
    return state, reward

def make_epsilon_greedy_policy(Q, epsilon, num_Actions):
    def policy_fn(observation):
        observation = tuple(observation)
        A = torch.ones(num_Actions, dtype=float) * epsilon / num_Actions
        best_action = torch.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn

def mc_control_on_policy_epsilon_greedy(env, num_episodes, discount=1.0, epsilon=0.1, epsilon_decay=0.9, device="cuda"):
    start_time = time.time()  # Start time for this section
    # Store the sum and number of returns for each state-action pair
    returns_sum = defaultdict(lambda: torch.tensor(0.0, device=device))
    returns_count = defaultdict(lambda: torch.tensor(0.0, device=device))

    # Q-value function initialized to zeros
    Q = defaultdict(lambda: torch.zeros(env.action_space.n, device=device))

    # Rewards
    y = torch.zeros(num_episodes, dtype=torch.float32, device=device)

    for i_episode in tqdm(range(num_episodes)):
        seen_roads = torch.zeros(len(roads)-1, device=device)
        
        # Epsilon-greedy policy
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        # Update epsilon
        epsilon = max(epsilon * epsilon_decay, 0.01)

        # Generate an episode
        episode = []
        state, _ = env.reset()
        state, _, _, _, _ = env.step(1)
        screen = env.render()

        state, reward = process_frame(state, roads, cars_top, cars_bottom, seen_roads, device)
        done = False
        total_reward = 0

        while not done:
            probs = policy(state)
            action = torch.multinomial(probs, 1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            screen = env.render()
            next_state, additional_reward = process_frame(next_state, roads, cars_top, cars_bottom, seen_roads, device)

            done = terminated or truncated
            episode.append((tuple(state), action, reward))
            total_reward += reward + additional_reward

            if done:
                break
            state = next_state

        y[i_episode] = total_reward

        # Process the episode in reverse order to calculate returns
        G = torch.tensor(0.0, device=device)
        for i in range(len(episode)-1, -1, -1):
            state, action, reward = episode[i]
            sa_pair = (state, action)
            
            G = reward + discount * G
            
            # Update the average return for the state-action pair
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

        # Debugging information
        if i_episode % 1 == 0 and i_episode > 0:
            print("\rEpisode {:8d}/{:8d} - Average reward {:.2f}".format(
                i_episode, num_episodes, torch.mean(y[max(0, i_episode-100):i_episode]).item()), end="")

    end_time = time.time()  # End time for the whole episode
    print(f"mc_control_on_policy_epsilon_greedy total time: {end_time - start_time:.4f} seconds")
    
    return Q, policy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make("ALE/Freeway-v5", difficulty=0, mode=2, obs_type="rgb", render_mode="rgb_array")
    env.reset()

    roads = [183, 167, 151, 135, 119, 103, 87, 71, 55, 39, 23]
    cars_bottom = torch.tensor(roads[:-1])-4
    cars_top = torch.tensor(roads[1:])+5

    Q, policy = mc_control_on_policy_epsilon_greedy(env, num_episodes=100, discount=1, epsilon=1, epsilon_decay=0.999, device=device)
