# Project_RL  

Welcome to **Project_RL**! This repository showcases our exploration of reinforcement learning techniques in three environments: Freeway, Tennis, and Pong. We implemented state-of-the-art algorithms, tested their performance, and analyzed their behavior.  

---

## 🚗 Freeways  
A classic Atari game where the agent controls a chicken trying to cross a multi-lane highway while avoiding cars.  

### 🛠️ What We Did  
- **Environment Preprocessing**: Resized, grayscaled, stacked frames, and normalized inputs for efficient model training.  
- **Algorithms**: Implemented **Deep Q-Networks (DQN)** with extensions like Double DQN, Dueling Networks, and Prioritized Replay Buffer. Also implemented **Reinforce**, a policy gradient algorithm, for comparison.  
- **Results**: Achieved the optimal reward of 21 using DQN with basic preprocessing and Reinforce with policy gradient optimization.  

![Freeway Gameplay](path/to/freeway.gif)  

### 📂 Files  
- **`Freeway.ipynb`**: Notebook demonstrating the preprocessing pipeline.  
- **`TabularMethod.py`**: Code for tabular methods (state transformations and Monte Carlo integration).  
- **`Rainbow_dwn.py`**: DQN implementation with extensions.  

---

## 🎾 Tennis  
A more complex Atari environment requiring the agent to learn hitting, positioning, and serving strategies in a tennis game.  

### 🛠️ What We Did  
- **Environment Simplification**: Cropped frames, extracted ball positions, and reduced the action space from 18 to 6 essential moves.  
- **Algorithms**: Implemented **PPO**, **Maskable PPO**, and **A2C**. Enhanced Maskable PPO with wrappers for action masking, ball tracking, and intermediate rewards.  
- **Results**: Maskable PPO significantly outperformed other models but could not win a full game due to environment complexity.  

![Tennis Gameplay](path/to/tennis.gif)  

### 📂 Files  
- **`SB3_Tennis_A2C.py`**: Train a single agent with A2C.  
- **`SB3_Tennis_ppo.py`**: Train a single agent with PPO.  
- **`SB3_TennisDetectPilotA2C.py`**: Train Maskable A2C with advanced wrappers.  
- **`SB3_TennisDetectPilotPPO.py`**: Train Maskable PPO with action masking and ball tracking.  

---

## 🏓 Pong  
Simulates a two-player table tennis game where agents compete to score points.  

### 🛠️ What We Did  
- **Single-Agent Approach**: Trained a right paddle agent using PPO in a simplified setup.  
- **Multi-Agent Approach**: Extended the Gym environment to PettingZoo, enabling two agents to train against each other.  
- **Challenges**: Despite promising setups, agents converged to suboptimal strategies, focusing on serving rather than rallying.  

![Pong Gameplay](path/to/pong.gif)  

### 📂 Files  
- **`MAS_tournament_ppo.py`**: Train two agents with PettingZoo and PPO.  
- **`MAS_tournament_dqn.py`**: Train two agents with DQN.  
- **`MAS_tournament_reinforce.py`**: Train two agents with Reinforce.  
- **`LoadModels.py`**: Utilities to load pre-trained models for inference.  
- **`single_agent_ppo.py`**: Train a single agent with PPO.  
- **`Config`.py**: Containing the config files
- **`wrappers.py`** Wrappers implementations

---

## 🚀 Highlights  
- **Algorithms Implemented**: DQN, Reinforce, PPO, Maskable PPO, A2C.  
- **Preprocessing Innovations**: Simplified observations using cropping, grayscale, red channel filtering, and frame stacking.  
- **Insights Gained**: Highlighted the strengths and limitations of value-based vs. policy gradient methods across varying complexities.  

---

## 📧 Questions or Contributions?  
Feel free to open an issue or submit a pull request for improvements or suggestions.  
