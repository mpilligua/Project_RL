import numpy as np

def step(self, net, epsilon=0.0, device="cpu"):
    done_reward = None
    done_number_steps = None
    def get_action(state):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_ = np.array([state])
            state = torch.tensor(state_).to(device)
            q_vals = net(state)
            _, act_ = torch.max(q_vals, dim=1)
            action = int(act_.item())

        return action
    # First Step
    action = get_action(self.current_state)
    new_state, reward, terminated, truncated, _ = self.env.step(action)
    is_done = terminated or truncated
    self.total_reward += reward
    
    self.number_of_frames_per_episode += 1
    if not is_done:  # Only take a second step if the first is not terminal
        next_action = get_action(new_state)
        next_new_state, next_reward, terminated, truncated, _ = self.env.step(next_action)
        next_done = terminated or truncated
        self.number_of_frames_per_episode += 1
    else:
        next_new_state = new_state
        next_reward = 0
        next_done = True
    
    combined_reward = reward + self.gamma * next_reward
    combined_done = is_done or next_done
    
    exp = Experience(self.current_state, action, combined_reward, combined_done, next_new_state)
    self.exp_replay_buffer.append(exp)
    
    # Update the current state, either to the one produced calling step one time or two times depending it the agent arrived to a terminal state or not
    self.current_state = next_new_state
    
    if is_done:
        done_reward = self.total_reward
        done_number_steps = self.number_of_frames_per_episode
        self._reset()
        
    
    return done_reward, done_number_steps