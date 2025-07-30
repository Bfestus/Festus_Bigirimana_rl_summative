from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os
import torch
import torch.nn as nn
import numpy as np
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

class REINFORCE:
    def __init__(self, env):
        self.env = env
        # Hyperparameters
        self.learning_rate = 5e-4    # Learning rate
        self.gamma = 0.99           # Discount factor
        self.n_steps = 2048         # Steps per update
        self.ent_coef = 0.01        # Entropy coefficient
        self.max_grad_norm = 0.5    # Gradient clipping
        self.batch_size = 64        # Batch size
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, env.action_space.n),
            nn.Softmax(dim=-1)
        )
        
        # Initialize network weights
        for layer in self.policy:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        probs = probs / probs.sum()
        
        action = torch.multinomial(probs, 1)
        return action.item(), probs[action].log()

    def update_policy(self, rewards, log_probs):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            loss = -log_prob * R
            policy_loss.append(loss)
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

def train_reinforce():
    print("Starting REINFORCE training...")
    env = EmergencyDroneEnv(render_mode=None)
    model = REINFORCE(env)
    num_episodes = 2000
    best_reward = -float('inf')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        rewards = []
        log_probs = []
        
        for t in range(200):  # Max steps per episode
            action, log_prob = model.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Enhanced reward shaping
            if next_state[3] > state[3]:  # Found survivor
                reward += 10.0
            
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
            
            if terminated or truncated:
                break
        
        # Update policy
        model.update_policy(rewards, log_probs)
        
        # Logging
        total_reward = sum(rewards)
        if episode % 10 == 0:
            print(f'Episode {episode}: Reward={total_reward:.2f}, '
                  f'Survivors={state[3]:.0f}/{state[4]:.0f}, Battery={state[2]:.1f}%')
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            os.makedirs("models/reinforce", exist_ok=True)
            torch.save(model.policy.state_dict(), 
                      "models/reinforce/reinforce_drone_best.pth")
    
    print("✅ Training completed!")

def evaluate_reinforce(model_path="models/reinforce/reinforce_drone_best.pth", 
                      episodes=3):
    print("\nEvaluating REINFORCE agent...")
    env = EmergencyDroneEnv(render_mode="human")
    model = REINFORCE(env)
    model.policy.load_state_dict(torch.load(model_path))
    model.policy.eval()
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        action_counts = {i: 0 for i in range(7)}
        step_count = 0
        
        print(f"\nEpisode {ep+1} starting with {state[4]:.0f} survivors")
        
        for step in range(200):
            step_count = step + 1
            action, _ = model.select_action(state)
            action_counts[action] += 1
            
            state, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            
            print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, "
                  f"Battery={state[2]:.1f}, Survivors={state[3]:.0f}/{state[4]:.0f}")
            
            if terminated or truncated:
                break
        
        # Episode summary
        print(f"\nEpisode {ep+1} Action Distribution:")
        actions = ["Move Up", "Move Down", "Move Left", "Move Right", 
                  "Scan Area", "Return to Base", "Emergency Land"]
        for i, name in enumerate(actions):
            print(f"{name}: {action_counts[i]} times ({action_counts[i]/step_count*100:.1f}%)")
        
        print(f"\nEpisode {ep+1} Summary:")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Survivors found: {state[3]:.0f}/{state[4]:.0f}")
        print(f"Final battery: {state[2]:.1f}%")
    
    env.close()

if __name__ == "__main__":
    # Uncomment only one at a time
    # train_reinforce()
    evaluate_reinforce()