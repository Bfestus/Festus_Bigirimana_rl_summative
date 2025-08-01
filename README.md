# Drone Rescue RL Summative

This project implements and compares four reinforcement learning algorithms (DQN, PPO, A2C, REINFORCE) for an autonomous drone rescue mission in a custom simulated environment. The drone agent must efficiently locate and assist survivors while managing battery life and avoiding obstacles.
## Environment visualization
Here is GIF:

![rescue_all_agent_demo](https://github.com/user-attachments/assets/d81054f3-9bcd-4435-bd1c-c393a2d113a1)

## Features
- Custom OpenAI Gym environment for drone rescue
- Discrete action and continuous state space
- Reward structure for survivor rescue and battery management
- Training scripts for DQN, PPO, A2C, and REINFORCE (using Stable Baselines3)
- Video recordings of agent performance
- Performance analysis and comparison plots

## Project Structure
```
environment/         # Custom environment and rendering
training/            # Training scripts for each algorithm
models/              # Saved trained models
videos/              # Agent performance videos
results/             # Performance metrics and plots
main.py              # Main evaluation and recording script
requirements.txt     # Dependencies
README.md            # Project overview (this file)
```

## Usage
1. Install dependencies:  
   `pip install -r requirements.txt`
2. Train or load models using scripts in `training/`
3. Run evaluation and record videos:  
   `python main.py`
4. View results in the `results/` and `videos/` folders

##
