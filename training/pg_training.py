from stable_baselines3 import PPO
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
        if len(self.locals["infos"]) > 0 and "episode" in self.locals["infos"][0]:
            ep_info = self.locals["infos"][0]["episode"]
            reward = ep_info["r"]
            self.episode_rewards.append(reward)
            print(f"Episode {len(self.episode_rewards)} reward: {reward:.2f}")
        return True

def train_ppo():
    print("Starting PPO training with optimized movement and battery management...")
    env = EmergencyDroneEnv(render_mode=None)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        # Learning parameters
        learning_rate=5e-5,        # Lower learning rate for stability
        n_steps=2048,              # Longer episodes for better learning
        batch_size=128,            # Larger batch size
        n_epochs=20,               # More training epochs
        gamma=0.99,                # Standard discount factor
        gae_lambda=0.95,           # GAE parameter
        clip_range=0.4,            # Higher clip range for more exploration
        ent_coef=0.1,             # Much higher entropy coefficient
        vf_coef=0.5,              # Balanced value function coefficient
        max_grad_norm=0.5,         # Standard gradient clipping
        # Deeper network architecture
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 256], # Deeper policy network
                vf=[256, 256, 256]  # Deeper value network
            ),
            activation_fn=nn.ReLU,
            ortho_init=True,
            log_std_init=-1.0      # Higher initial exploration
        ),
        tensorboard_log="./logs/pg/"
    )
    
    # Train for longer with progress monitoring
    callback = RewardLoggerCallback()
    model.learn(
        total_timesteps=1000000,   # Much longer training time
        callback=callback,
        progress_bar=True
    )
    
    os.makedirs("models/pg", exist_ok=True)
    model.save("models/pg/ppo_drone_final")
    print("✅ Training completed! Model saved to models/pg/ppo_drone_final")

def evaluate_ppo(model_path="models/pg/ppo_drone_final.zip", episodes=3):
    print("\nEvaluating trained PPO agent...")
    env = EmergencyDroneEnv(render_mode="human")
    model = PPO.load(model_path)
    
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    total_survivors_found = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        print(f"\nEpisode {ep+1} starting with {obs[4]:.0f} survivors")
        
        for i in range(200):
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)
            action_counts[action_int] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            
            print(f"Step {i+1}: Action={action_int}, Reward={reward:.2f}, "
                  f"Battery={obs[2]:.1f}, Survivors={obs[3]:.0f}/{obs[4]:.0f}, "
                  f"Position=({obs[0]:.0f},{obs[1]:.0f})")
            
            if terminated or truncated:
                total_survivors_found += obs[3]
                print(f"Episode ended: {info.get('termination_reason', 'unknown')}")
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
        
        print(f"Episode {ep+1} Summary:")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Survivors found: {obs[3]:.0f}/10")
        print(f"Final battery: {obs[2]:.1f}%")
    
    env.close()

if __name__ == "__main__":
    # Uncomment only one at a time
    # train_ppo()
    evaluate_ppo()