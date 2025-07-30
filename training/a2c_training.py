from stable_baselines3 import A2C
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

def train_a2c():
    print("Starting A2C training with optimized parameters...")
    env = EmergencyDroneEnv(render_mode=None)
    
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        # Learning parameters
        learning_rate=5e-4,        # Balanced learning rate
        n_steps=128,               # Much longer episodes
        gamma=0.99,                # Standard discount
        gae_lambda=0.95,           # Standard GAE
        ent_coef=0.2,             # Very high entropy for exploration
        vf_coef=0.6,              # Balanced value learning
        max_grad_norm=0.8,         # Relaxed gradient clipping
        use_rms_prop=False,        # Switch to Adam optimizer
        normalize_advantage=True,
        # Enhanced network architecture
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 128, 128], # Wider then deeper policy
                vf=[256, 128, 128]  # Matching value network
            ),
            activation_fn=nn.ReLU,
            ortho_init=True,
            log_std_init=-0.5      # Much higher initial exploration
        ),
        tensorboard_log="./logs/a2c/"
    )
    
    callback = RewardLoggerCallback()
    # Increased training time
    model.learn(
        total_timesteps=1200000,   # 50% more training time
        callback=callback,
        progress_bar=True
    )
    
    os.makedirs("models/a2c", exist_ok=True)
    model.save("models/a2c/a2c_drone_final")
    print("✅ Training completed! Model saved to models/a2c/a2c_drone_final")

def evaluate_a2c(model_path="models/a2c/a2c_drone_final.zip", episodes=3):
    print("\nEvaluating trained A2C agent...")
    env = EmergencyDroneEnv(render_mode="human")
    model = A2C.load(model_path)
    
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    total_survivors_found = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        episode_actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        
        for i in range(200):
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)
            action_counts[action_int] += 1
            episode_actions[action_int] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            
            print(f"Step {i+1}: Action={action_int}, Reward={reward:.2f}, "
                  f"Battery={obs[2]:.1f}, Survivors={obs[3]:.0f}/{obs[4]:.0f}, "
                  f"Position=({obs[0]:.0f},{obs[1]:.0f})")
            
            if terminated or truncated:
                total_survivors_found += obs[3]
                break
        
        # Episode summary
        print(f"\nEpisode {ep+1} Action Distribution:")
        for act, count in episode_actions.items():
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
        
        print(f"\nEpisode {ep+1} Summary:")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Survivors found: {obs[3]:.0f}/{obs[4]:.0f}")
        print(f"Final battery: {obs[2]:.1f}%")
    
    env.close()

if __name__ == "__main__":
    # First train
    # train_a2c()
    # Then evaluate
    evaluate_a2c()