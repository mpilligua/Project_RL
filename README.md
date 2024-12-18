# Project_RL

# Explanation of files found in the repository
## Freeways  
- **`Freeway.ipynb`**: A Jupyter Notebook demonstrating how the transformations for tabular methods are applied.  
- **`TabularMethod.py`**: Code that integrates all the transformations for tabular methods with Monte Carlo techniques.  
- **`Rainbow_dwn.py`**: Code to execute DQN with extensions.  

## Tennis
    - Tennis/SB3_Tennis_A2C.py: Baseline PPO
    - Tennis/SB3_TennisDetectPilotA2C.py: Baseline A2C
    - Tennis/SB3_TennisDetectPilotPPO.py: Extended wrappers and Maskable A2C  
    - Tennis/SB3_TennisDetectPilotRNN.py: Extended wrappers and Maskable PPO
    - Tennis/SB3_Tennis_ppo.py

## Pong  
- **`MAS_tournament_ppo.py`**: Code to train two agents in a multi-environment setup using PettingZoo and a custom implementation of PPO.  
- **`MAS_tournament_dqn.py`**: Code to train two agents in a multi-environment setup using PettingZoo and a custom implementation of DQN.  
- **`MAS_tournament_dqn_utils.py`**: Utility functions for executing `MAS_tournament_dqn.py`, including the DQN model and its configuration.  
- **`MAS_tournament_reinforce.py`**: Code to train two agents in a multi-environment setup using PettingZoo and a custom implementation of Reinforce.  
- **`MAS_tournament_reinforce_against_trained.py`**: Code to train two agents in a multi-environment setup using PettingZoo and a custom implementation of Reinforce, with additional functionality to load a PPO-trained single agent.  
- **`single_agent_ppo.py`**: Code to train a single agent in a single-environment setup using Stable-Baselines PPO and a Gymnasium environment.  
