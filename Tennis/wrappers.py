import gymnasium as gym
import numpy as np
import collections
import torch

from SB3_Tennis import log
import cv2

def is_ball_color(color):
    """
    Determines if a given RGB color belongs to the ball or the field.

    Parameters:
        color (tuple): RGB color to check, e.g., (R, G, B).

    Returns:
        bool: True if the color is identified as a ball color, False otherwise.
    """
    ball_colors = [
        (137, 179, 156),  # Ball color 1
        (183,205,193)   # Ball color 2
    ]
    field_colors = [
        (45, 126, 82),    # Green
        (147, 127, 100),  # Red
        (120, 128, 224),   # Blue
        (147, 127, 100),
    ]

    def euclidean_distance(c1, c2):
        return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))
    
    print("The shape of the color is", color.shape)
    print("The shape of bc is", np.array(ball_colors[0]).shape)
    # Threshold for deciding similarity (tuned empirically or as needed)
    threshold = 50

    # Compute distances to ball colors and field colors
    ball_distances = [euclidean_distance(color, bc) for bc in ball_colors]
    field_distances = [euclidean_distance(color, fc) for fc in field_colors]

    # Check if the color is closer to any ball color than to field colors
    return min(ball_distances) < min(field_distances) and min(ball_distances) < threshold


class GetBallPositionAndHuman(gym.ObservationWrapper):
    """
    Processes the environment observation to identify ball position, human position, 
    and the other player's position using color-based filtering.

    Observations are masked using a predefined binary mask, and specific pixels are 
    analyzed for ball-like and human-like colors to track their positions.
    """
    def __init__(self, env):
        super().__init__(env)

        self.last_ball_position = (0, 0)
        self.last_human_position = (0, 0)
        self.other_player_position = (0,0)
        
        self.maskara = cv2.imread('/ghome/mpilligua/RL/Project_RL/Tennis/MariaMaskara2.jpg')
        self.maskara = self.maskara.mean(axis=-1)
        _, self.maskara = cv2.threshold(self.maskara, 127, 255, cv2.THRESH_BINARY)
        #self.maskara = self.maskara[np.newaxis, :, :]
        self.maskara = np.repeat(self.maskara[np.newaxis, :, :], 3, axis=0)
        
        # Define the observation space as a Dict

        self.observation_space = gym.spaces.Dict({
            "ball_position": gym.spaces.Box(low=0, high=80, shape=(1, 2), dtype=np.float32),
            "masked_observation": gym.spaces.Box(
                low=0,
                high=1,
                shape=(
                    79,#env.observation_space.shape[1],
                    78#env.observation_space.shape[2],
                ),
                dtype=np.float32,
            ),
            "human_position": gym.spaces.Box(low=0, high=80, shape=(1, 2), dtype=np.float32),
            "other_player_position": gym.spaces.Box(low=0, high=80, shape=(1, 2), dtype=np.float32),
            "server": gym.spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.float32)
        })
    def observation(self, observation):
        # Apply the mask to the observation
        
        obs = observation * self.maskara  # Ensure maskara has the same shape as observation
        obs /= 255.0
        
    
        # Get image dimensions (assuming channel-first: (channels, height, width))
        channels, height, width = obs.shape

        # Reshape the image for easier processing (flatten to [num_pixels, 3])
        flattened_obs = obs.transpose(1, 2, 0).reshape(-1, 3)  # Convert to (height, width, channels) first

        # Define ball colors and field colors as numpy arrays for vectorized operations
        ball_colors = np.array([
            [137, 179, 156],  # Ball color 1
            [183, 205, 193],  # Ball color 2
        ])
        field_colors = np.array([
            [45, 126, 82],     # Field green
            [147, 127, 100],   # Field red
            [120, 128, 224],   # Field blue
            [147, 127, 100],   # Field brown
        ])
        
        human_colors = np.array(
            [244,132,132]        
        )
        
        other_player_colors = np.array([116,132,244])

        # Calculate Euclidean distances from each pixel to ball and field colors
        ball_distances = np.min(np.linalg.norm(flattened_obs[:, None, :] - ball_colors[None, :, :], axis=2), axis=1)
        field_distances = np.min(np.linalg.norm(flattened_obs[:, None, :] - field_colors[None, :, :], axis=2), axis=1)
        human_distances = np.min(np.linalg.norm(flattened_obs[:, None, :] - human_colors[None, :], axis=2), axis=1)
        other_player = np.min(np.linalg.norm(flattened_obs[:, None, :] - other_player_colors[None, :], axis=2), axis=1)
        
        # Determine which pixels are ball-like
        threshold = 50
        is_ball_pixel = (ball_distances < field_distances) & (ball_distances < threshold)
        threshold_human = 10
        is_human_pixel = (human_distances < field_distances) & (human_distances < threshold_human)
        is_other_player = (other_player < field_distances) & (other_player < threshold_human)
        
        # Find the coordinates of ball-like pixels
        ball_pixel_indices = np.where(is_ball_pixel)[0]
        if ball_pixel_indices.size == 0:
            # If no ball pixel is found, return the last known position
            ball_coordinates = self.last_ball_position
        else:
            # Convert the flattened index back to (y, x) coordinates
            ball_coordinates = np.unravel_index(ball_pixel_indices[0], (height, width))
            self.last_ball_position = ball_coordinates

        
        human_pixel_indices = np.where(is_human_pixel)[0]
        if human_pixel_indices.size == 0:
            # If no ball pixel is found, return the last known position
            human_coordinates = self.last_human_position
        else:
            # Convert the flattened index back to (y, x) coordinates
            human_coordinates = np.unravel_index(human_pixel_indices[0], (height, width))
            self.last_human_position = human_coordinates

        other_player_indices = np.where(is_other_player)[0]
        if other_player_indices.size == 0:
            # If no ball pixel is found, return the last known position
            other_player_coordinates = self.other_player_position
        else:
            # Convert the flattened index back to (y, x) coordinates
            other_player_coordinates = np.unravel_index(other_player_indices[0], (height, width))
            self.other_player_position = other_player_coordinates

        #print(ball_coordinates)
        # Afegir la red dim i pasar la imatge normalizada

        red_channel = obs[0]
        
        self.env.server
        
        
        y, x = ball_coordinates
        ball_coordinates = (y/height, x/width)
        y, x = human_coordinates
        human_coordinates = (y/height, x/width)
        y, x = other_player_coordinates
        other_player_coordinates = (y/height, x/width)
        
        

        # Return the processed observation and the ball position
        return {
            "masked_observation": red_channel.astype(np.float32) / 255.0,
            "ball_position": np.array(ball_coordinates, dtype=np.float32).reshape(1, 2),
            "human_position": np.array(human_coordinates, dtype=np.float32).reshape(1, 2),
            "other_player_position": np.array(other_player_coordinates, dtype=np.float32).reshape(1, 2),
            "server": np.array([self.env.server], dtype=np.float32).reshape(1, 1)
        }


class GetBallPosition(gym.ObservationWrapper):
    """
    Processes the environment observation to identify and track the position of the ball.

    It applies a binary mask to the input observation and filters ball-like pixels 
    based on their color similarity to predefined ball colors.
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_ball_position = (0, 0)
        self.maskara = cv2.imread('/ghome/mpilligua/RL/Project_RL/Tennis/MariaMaskara2.jpg')
        self.maskara = self.maskara.mean(axis=-1)
        _, self.maskara = cv2.threshold(self.maskara, 127, 255, cv2.THRESH_BINARY)
        #self.maskara = self.maskara[np.newaxis, :, :]
        self.maskara = np.repeat(self.maskara[np.newaxis, :, :], 3, axis=0)
        
        # Define the observation space as a Dict

        self.observation_space = gym.spaces.Dict({
            "ball_position": gym.spaces.Box(low=0, high=80, shape=(1, 2), dtype=np.float32),
            "masked_observation": gym.spaces.Box(
                low=0,
                high=1,
                shape=(
                    79,#env.observation_space.shape[1],
                    78#env.observation_space.shape[2],
                ),
                dtype=np.float32,
            ),
        })
    def observation(self, observation):
        # Apply the mask to the observation
        
        obs = observation * self.maskara  # Ensure maskara has the same shape as observation
        obs /= 255.0
        
    
        # Get image dimensions (assuming channel-first: (channels, height, width))
        channels, height, width = obs.shape

        # Reshape the image for easier processing (flatten to [num_pixels, 3])
        flattened_obs = obs.transpose(1, 2, 0).reshape(-1, 3)  # Convert to (height, width, channels) first

        # Define ball colors and field colors as numpy arrays for vectorized operations
        ball_colors = np.array([
            [137, 179, 156],  # Ball color 1
            [183, 205, 193],  # Ball color 2
        ])
        field_colors = np.array([
            [45, 126, 82],     # Field green
            [147, 127, 100],   # Field red
            [120, 128, 224],   # Field blue
            [147, 127, 100],   # Field brown
        ])

        # Calculate Euclidean distances from each pixel to ball and field colors
        ball_distances = np.min(np.linalg.norm(flattened_obs[:, None, :] - ball_colors[None, :, :], axis=2), axis=1)
        field_distances = np.min(np.linalg.norm(flattened_obs[:, None, :] - field_colors[None, :, :], axis=2), axis=1)

        # Determine which pixels are ball-like
        threshold = 50
        is_ball_pixel = (ball_distances < field_distances) & (ball_distances < threshold)

        # Find the coordinates of ball-like pixels
        ball_pixel_indices = np.where(is_ball_pixel)[0]
        if ball_pixel_indices.size == 0:
            # If no ball pixel is found, return the last known position
            ball_coordinates = self.last_ball_position
        else:
            # Convert the flattened index back to (y, x) coordinates
            ball_coordinates = np.unravel_index(ball_pixel_indices[0], (height, width))
            self.last_ball_position = ball_coordinates


        #print(ball_coordinates)
        # Afegir la red dim i pasar la imatge normalizada

        red_channel = obs[0]

        # Return the processed observation and the ball position
        return {
            "masked_observation": red_channel.astype(np.float32) / 255.0,
            "ball_position": np.array(ball_coordinates, dtype=np.float32).reshape(1, 2),
        }

    #def observation(self, observation):
    #    # Apply the mask to the observation
    #    obs = observation * self.maskara
    #    print("Observation shape", obs.shape)
#
    #    # Find ball coordinates
    #    ball_coordinates = np.column_stack(np.where(np.apply_along_axis(is_ball_color, -1, obs))).T
    #    if len(ball_coordinates) == 0:
    #        ball_coordinates = self.last_ball_position
    #    else:
    #        self.last_ball_position = ball_coordinates[0]
#
    #    # Return as a dictionary
    #    return {
    #        "ball_position": np.array(self.last_ball_position, dtype=np.float32).reshape(1, 1),
    #        "masked_observation": obs.astype(np.float32)
    #     }

class GetBallPosition2(gym.ObservationWrapper):
    """
    A variant of ball position tracking wrapper that masks the input observation 
    and uses a more efficient approach to extract ball coordinates.
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_ball_position = (0, 0)
        self.maskara = cv2.imread('/ghome/mpilligua/RL/Project_RL/Tennis/MariaMaskara2.jpg')
        self.maskara = self.maskara.mean(axis=-1)
        _, self.maskara = cv2.threshold(self.maskara, 127, 255, cv2.THRESH_BINARY, self.maskara)
        self.maskara = np.repeat(self.maskara[np.newaxis, :, :], 3, axis=0)
        
        self.observation_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=80, shape=(1, 1), dtype=np.float32),   
                                                   gym.spaces.Box(low=0, high=1, shape=(env.observation_space.shape[0], 
                                                                                        env.observation_space.shape[1],
                                                                                        env.observation_space.shape[2]), dtype=np.float32)))
        
    def observation(self, observation):


        obs = observation[0] * self.maskara
        
        

        ball_coordinates = np.column_stack(np.where(np.apply_along_axis(is_ball_color, -1, obs))).T
        if len(ball_coordinates) == 0:
            ball_coordinates = self.last_ball_position
        else:
            self.last_ball_position = ball_coordinates[0]
            
        return obs, self.last_ball_position


class ActionPenaltyWrapper(gym.RewardWrapper):
    """
    Adds a penalty to the reward when the same action (e.g., action 0) is repeated 
    consecutively for a specified number of steps.
    """
    def __init__(self, env, max_consecutive_zeros=5, penalty_reward=-10.0):
        super().__init__(env)
        self.max_consecutive_zeros = max_consecutive_zeros
        self.penalty_reward = penalty_reward
        self.consecutive_zeros = 0

    def reset(self, **kwargs):
        self.consecutive_zeros = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        if action == 0:
            self.consecutive_zeros += 1
        else:
            self.consecutive_zeros = 0
        
        if self.consecutive_zeros > self.max_consecutive_zeros:
            reward += self.penalty_reward  # Apply penalty
        
        return obs, reward, terminated, truncated, info



    
class ImageToPyTorch(gym.ObservationWrapper):
    """
    Converts the image observation from (height, width, channels) format to 
    (channels, height, width) format, suitable for PyTorch input.
    """
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Scales the pixel values of the input observation to the range [0, 1].
    """
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class RewardLongPoints(gym.RewardWrapper):
    """
    Modifies the reward system to provide a positive reward for prolonged points. 
    The longer the agent avoids losing a point, the higher the reward.
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
            if self.frames < 50:
                added_reward = -2
            else:
                added_reward = min(self.frames, 150) / (150*2)
            log(f"Added reward: {added_reward}")
            reward += added_reward
            self.frames = 0
        else:
            self.frames += 1

        self.last_reward = reward
        #if self.return_added_reward:
        #    return state, reward, added_reward, terminated, truncated, info
        #else:
        #    return state, reward, terminated, truncated, info
        return state, reward, terminated, truncated, info


class MapActions(gym.ActionWrapper):
    """
    Restricts the action space to 6 discrete actions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(6)

    def action(self, action):
        return action




class EndWhenLosePoint(gym.RewardWrapper):
    """
    Ends the episode and modifies the reward when the agent loses a point. 
    The reward depends on the duration of the point.
    """
    def __init__(self, env, do_eval):
        super().__init__(env)
        self.frames = 0
        self.last_reward = 0
        self.do_eval = do_eval
        
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.frames += 1
        
        if reward > self.last_reward:
            reward += 100
            self.last_reward = reward
            print(f"Reward: {reward}")
            self.frames = 0
            
            if self.do_eval:
                return state, reward, terminated, truncated, info
            
            return state, reward, True, truncated, info
        
        if reward < 0:
            if self.frames < 50:
                reward = -10
            elif self.frames > 150:
                reward = -50
            else:
                reward = min([self.frames/30, 5]) 
            # print(f"Reward: {reward}")
            self.frames = 0
            
            if self.do_eval:
                return state, reward, terminated, truncated, info
            
            return state, reward, True, truncated, info

        self.last_reward = reward
        # print(f"Reward: {reward}")
        return state, reward, terminated, truncated, info



class CropObservation(gym.ObservationWrapper):
    """
    Crops the input observation to a specified region defined by top, left, height, and width.
    """
    def __init__(self, env, top, left, height, width):
        super().__init__(env)
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(env.observation_space.shape[0], height, width), dtype=np.uint8)

    def observation(self, obs):
        #print(obs.shape)
        #print(obs[:, self.top:self.top+self.height, self.left:self.left+self.width].shape)
        return obs[:, self.top:self.top+self.height, self.left:self.left+self.width]


class ForceDifferentAction(gym.RewardWrapper):
    """
    Penalizes the agent for repeating the same action consecutively for a 
    predefined number of steps.
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
        
class Keep_red_dim(gym.ObservationWrapper):
    """
    Keeps only the red channel dimension of the input observation, removing other channels.
    """
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, old_shape[1], old_shape[2]), dtype=np.float32)
                                                
    def observation(self, observation):
        # print(observation.shape)
        return observation[:1]