import numpy as np
import torch
from collections import namedtuple, deque
from copy import deepcopy

import collections



class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'combined_reward', 'combined_done', 'next_new_state'])

    def __len__(self):
        return len(self.buffer)

    def append(self, current_state, action, combined_reward, combined_done, next_new_state):
        self.buffer.append(self.Experience(current_state, action, combined_reward, combined_done, next_new_state))

    def sample(self, BATCH_SIZE):
        indices = np.random.choice(len(self.buffer), BATCH_SIZE, replace=False)
        states, actions, combined_rewards, combined_dones, next_new_state = zip(*[self.buffer[idx] for idx in indices])
        
        return np.array(states), np.array(actions), np.array(combined_rewards, dtype=np.float32), \
               np.array(combined_dones, dtype=np.uint8), np.array(next_new_state)
          

class ExpGrowthFactor:
    def __init__(self, epsilon, max_epsilon, growth_rate):
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.growth_rate = growth_rate

    def update(self):
        self.epsilon = min(self.max_epsilon, self.epsilon * self.growth_rate)
        return self.epsilon
    
class PrioritizedExperienceReplayBuffer:
    def __init__(self, memory_size=50000, burn_in=10000, alpha=0.6, small_constant=0.05, growth_rate = 1.0005, beta=0.4):
        self.memory_size = memory_size
        self.beta = beta
        self.burn_in = burn_in
        self.small_constant = small_constant

        # Named tuple for storing experiences
        self.buffer = namedtuple('Buffer', field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)
        self.sampling_probabilities = np.zeros(memory_size, dtype=np.float32)

        self.alpha_factor = ExpGrowthFactor(max_epsilon=1, epsilon=alpha, growth_rate=growth_rate)
        self.beta_factor = ExpGrowthFactor(max_epsilon=1, epsilon=beta, growth_rate=growth_rate)

    def sample_batch(self, batch_size=32):
        if len(self.replay_memory) == 0:
            raise ValueError("Replay memory is empty. Cannot sample batch.")

        ps = self.sampling_probabilities[:len(self.replay_memory)]
        sampling_probs = ps**self.alpha_factor.epsilon / np.sum(ps**self.alpha_factor.epsilon)

        idxs = np.random.choice(np.arange(len(ps)),
                                 size=batch_size,
                                 replace=True,
                                 p=sampling_probs)
        experiences = [self.replay_memory[i] for i in idxs]

        weights = (len(self.replay_memory) * sampling_probs[idxs])**-self.beta_factor.epsilon
        normalized_weight = weights / weights.max()

        return idxs, experiences, normalized_weight

    def append(self, state, action, reward, done, next_state):
        priority = 1.0 if len(self.replay_memory) == 0 else self.sampling_probabilities.max()
        if len(self.replay_memory) == self.replay_memory.maxlen:
            if priority > self.sampling_probabilities.min():
                idx = self.sampling_probabilities.argmin()
                self.sampling_probabilities[idx] = priority
                self.replay_memory[idx] = self.buffer(state, action, reward, done, next_state)
        else:
            self.sampling_probabilities[len(self.replay_memory)] = priority
            self.replay_memory.append(self.buffer(state, action, reward, done, next_state))

    def update_priorities(self, idxs, priorities):
        for i, idx in enumerate(idxs):
            self.sampling_probabilities[idx] = max(priorities[i], self.small_constant)

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in