from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import EmergencyDroneEnv

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Log episode rewards if available in infos
        if len(self.locals["infos"]) > 0 and "episode" in self.locals["infos"][0]:
            ep_info = self.locals["infos"][0]["episode"]
            reward = ep_info["r"]
            self.episode_rewards.append(reward)
            print(f"Episode {len(self.episode_rewards)} reward: {reward:.2f}")
        return True

def train_dqn():
    print("Starting DQN training with enhanced survivor-finding parameters...")
    env = EmergencyDroneEnv(render_mode=None)
    
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        # Learning parameters - optimized for survivor finding
        learning_rate=1e-4,        # Reduced for more stable learning
        buffer_size=200000,        # Doubled buffer size
        learning_starts=5000,      # Earlier learning start
        batch_size=128,            # Balanced batch size
        gamma=0.99,                # Increased future reward importance
        train_freq=4,              # More frequent training
        gradient_steps=4,          # More gradient steps
        target_update_interval=1000,# More frequent target updates
        # Exploration parameters - focused on finding survivors
        exploration_fraction=0.3,   # Extended exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05, # Lower final exploration
        # Network parameters - deeper network
        policy_kwargs=dict(
            net_arch=[256, 256, 256, 128], # Deeper network
            activation_fn=nn.ReLU
        ),
        tensorboard_log="./logs/dqn/"
    )
    
    # Extended training time
    callback = RewardLoggerCallback()
    model.learn(
        total_timesteps=500000,    # Significantly more training time
        callback=callback,
        log_interval=100
    )
    
    os.makedirs("models/dqn", exist_ok=True)
    model.save("models/dqn/dqn_drone_final")
    print("✅ Training completed! Model saved to models/dqn/dqn_drone_final")

def evaluate_dqn(model_path="models/dqn/dqn_drone_final.zip", episodes=3):
    print("\nEvaluating trained DQN agent with 10 survivors...")
    env = EmergencyDroneEnv(render_mode="human")
    model = DQN.load(model_path)
    
    # Convert numpy array action to integer for counting
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    total_survivors_found = 0
    last_position = None
    stuck_counter = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        print(f"\nEpisode {ep+1} starting with {obs[4]:.0f} survivors")
        
        for i in range(200):
            action, _ = model.predict(obs, deterministic=True)
            # Convert numpy array to integer
            action_int = int(action)
            action_counts[action_int] += 1  # Count actions
            
            # Check if stuck
            current_position = (obs[0], obs[1])
            if current_position == last_position:
                stuck_counter += 1
                if stuck_counter > 10:
                    print("⚠️ Agent appears stuck! Encouraging exploration...")
                    action = env.action_space.sample()
            else:
                stuck_counter = 0
            last_position = current_position

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            
            print(f"Step {i+1}: Action={action_int}, Reward={reward:.2f}, "
                  f"Battery={obs[2]:.1f}, Survivors={obs[3]:.0f}/{obs[4]:.0f}, "
                  f"Position=({obs[0]:.0f},{obs[1]:.0f})")
            
            if terminated or truncated:
                total_survivors_found += obs[3]
                break
        
        # Episode summary with action distribution
        print(f"\nEpisode {ep+1} Action Distribution:")
        for act, count in action_counts.items():
            action_name = {
                0: "Move Up",
                1: "Move Down",
                2: "Move Left",
                3: "Move Right",
                4: "Scan Area",
                5: "Return to Base",
                6: "Emergency Land"
            }[act]
            print(f"{action_name}: {count} times ({count/(i+1)*100:.1f}%)")
        
        print(f"Total steps: {i+1}")
        print(f"Survivors found: {obs[3]:.0f}/10")
        print(f"Final battery: {obs[2]:.1f}%")
    
    env.close()

if __name__ == "__main__":
    # First train with new parameters
    # train_dqn()
    # Then evaluate
    evaluate_dqn()
    pass