# Project_RL  

Welcome to **Project_RL**! This repository contains code for various reinforcement learning projects, focusing on different environments and techniques. Below is an explanation of the files available in the repository.  

---

## 📂 Freeways  
Files related to the **Freeway** environment:  
- **`Freeway.ipynb`**  
  - A Jupyter Notebook demonstrating how transformations for tabular methods are applied.  
- **`TabularMethod.py`**  
  - Code that integrates all the transformations for tabular methods with Monte Carlo techniques.  
- **`Rainbow_dwn.py`**  
  - Code to execute DQN with extensions.  

---

## 🎾 Tennis  
Files related to the **Tennis** environment:  
- **`SB3_Tennis_A2C.py`**  
  - Code to train a single agent in a tennis environment using Stable-Baselines3 and the A2C algorithm.  
- **`SB3_Tennis_ppo.py`**  
  - Code to train a single agent in a tennis environment using Stable-Baselines3 and the PPO algorithm.  
- **`SB3_TennisDetectPilotA2C.py`**  
  - Code to train Maskable A2C using additional wrappers.  
- **`SB3_TennisDetectPilotPPO.py`**  
  - Code to train Maskable PPO using additional wrappers.
- **`Config`**
  - .py containing the config files
- **`wrappers.py`**
  - Wrappers implementations

---

## 🕹️ Pong  
Files related to the **Pong** environment:  
- **`MAS_tournament_ppo.py`**  
  - Code to train two agents in a multi-environment setup using PettingZoo and a custom implementation of PPO.  
- **`MAS_tournament_dqn.py`**  
  - Code to train two agents in a multi-environment setup using PettingZoo and a custom implementation of DQN.  
- **`MAS_tournament_dqn_utils.py`**  
  - Utility functions for executing `MAS_tournament_dqn.py`, including the DQN model and its configuration.  
- **`MAS_tournament_reinforce.py`**  
  - Code to train two agents in a multi-environment setup using PettingZoo and a custom implementation of Reinforce.  
- **`MAS_tournament_reinforce_against_trained.py`**  
  - Code to train two agents in a multi-environment setup using PettingZoo and a custom implementation of Reinforce, with additional functionality to load a PPO-trained single agent.  
- **`single_agent_ppo.py`**  
  - Code to train a single agent in a single-environment setup using Stable-Baselines PPO and a Gymnasium environment.  
- **`LoadModels.py`**  
  - Contains all necessary functions and utilities to load models for inference.  
  - **Weights**: Pre-trained model weights are available in the repository under the names: `ppo_weight_right` and `ppo_weight_left`.  

---

## 🚀 Getting Started  
To get started, explore the files in the relevant sections above and run the scripts according to your use case. For detailed implementation details or configurations, refer to the code comments within each file.  

---

## 📧 Questions or Contributions?  
Feel free to open an issue or submit a pull request for any improvements or suggestions.  
