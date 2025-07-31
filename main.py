import sys
import os
import time
import torch
import numpy as np
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from training.reinforce_training import REINFORCE
from environment.custom_env import EmergencyDroneEnv

class DroneRescueEvaluator:
    def __init__(self):
        """Initialize evaluator with environment and model paths"""
        self.env = EmergencyDroneEnv(render_mode="rgb_array")
        self.videos_dir = "videos"
        self.results_dir = "results"
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define paths for trained models
        self.models = {
            "DQN": "models/dqn/dqn_drone_final.zip",
            "PPO": "models/pg/ppo_drone_final.zip",
            "A2C": "models/a2c/a2c_drone_final.zip",
            "REINFORCE": "models/reinforce/reinforce_drone_best.pth"
        }

    def load_model(self, algo_name):
        """Load model with proper handling for each algorithm"""
        try:
            if algo_name == "REINFORCE":
                model = REINFORCE(self.env)
                model.policy.load_state_dict(torch.load(self.models[algo_name]))
                model.policy.eval()
                return model
            else:
                model_class = {"DQN": DQN, "PPO": PPO, "A2C": A2C}[algo_name]
                return model_class.load(self.models[algo_name])
        except Exception as e:
            print(f"❌ Error loading {algo_name}: {str(e)}")
            return None

    def get_action(self, model, state, algo_name):
        """Get action from model with proper handling for each algorithm"""
        try:
            if algo_name == "REINFORCE":
                state_tensor = torch.FloatTensor(state)
                action, _ = model.select_action(state_tensor)
                return action
            else:
                return model.predict(state, deterministic=True)[0]
        except Exception as e:
            print(f"❌ Error getting action for {algo_name}: {str(e)}")
            return None

    def record_algorithm(self, algo_name, episodes=3):
        """Record algorithm performance with minimal overlays and environment-matching colors"""
        print(f"\n🎥 Recording {algo_name} performance...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.videos_dir, f"{algo_name}_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = None
        fps = 30
        
        try:
            model = self.load_model(algo_name)
            if model is None:
                return None
            
            total_survivors = 0
            for ep in range(episodes):
                state = self.env.reset()[0]
                episode_reward = 0
                survivors_found = 0
                
                print(f"\n📽️ Recording Episode {ep+1}/{episodes}")
                
                while True:
                    frame = self.env.render()
                    if frame is not None:
                        # Convert RGB (pygame) to BGR (cv2)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        h, w = frame_bgr.shape[:2]
                        if video is None:
                            video = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                        video.write(frame_bgr)
                    
                    action = self.get_action(model, state, algo_name)
                    if action is None:
                        raise Exception("Failed to get action")
                    
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    episode_reward += reward
                    
                    if next_state[3] > state[3]:
                        survivors_found += 1
                        # No overlays, just raw environment visuals
                    
                    state = next_state
                    
                    if done or truncated:
                        total_survivors += survivors_found
                        break
                
                time.sleep(0.5)
            
            if video is not None:
                video.release()
            print(f"✅ Video saved: {video_path}")
            print(f"Total survivors found: {total_survivors}")
            
        except Exception as e:
            print(f"❌ Error recording {algo_name}: {str(e)}")
            if video is not None:
                video.release()
            return None

    def analyze_performance(self, episodes=5):
        """Generate comprehensive performance analysis"""
        metrics = {algo: {
            "survivors_found": [],
            "episode_rewards": [],
            "battery_efficiency": [],
            "completion_time": [],
            "steps_taken": []
        } for algo in ["DQN", "PPO", "A2C", "REINFORCE"]}
        
        print("\n📊 Analyzing Algorithm Performance")
        print("=" * 50)
        
        for algo in metrics.keys():
            print(f"\nEvaluating {algo}...")
            model = self.load_model(algo)
            if model is None:
                continue
            
            for episode in range(episodes):
                state = self.env.reset()[0]
                ep_reward = 0
                start_time = time.time()
                steps = 0
                
                while True:
                    action = self.get_action(model, state, algo)
                    if action is None:
                        break
                        
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    ep_reward += reward
                    steps += 1

                    if done or truncated:
                        metrics[algo]["survivors_found"].append(next_state[3])
                        metrics[algo]["episode_rewards"].append(ep_reward)
                        metrics[algo]["battery_efficiency"].append(next_state[2])
                        metrics[algo]["completion_time"].append(time.time() - start_time)
                        metrics[algo]["steps_taken"].append(steps)
                        break
                    
                    state = next_state
        
        self._save_performance_metrics(metrics)
        self._plot_performance_comparison(metrics)
        return metrics

    def _save_performance_metrics(self, metrics):
        """Save detailed performance metrics to markdown"""
        with open(os.path.join(self.results_dir, "performance_metrics.md"), "w") as f:
            f.write("# Algorithm Performance Analysis\n\n")
            
            # Summary table
            f.write("## Performance Summary\n\n")
            f.write("| Algorithm | Survivors | Reward | Battery | Time (s) | Steps |\n")
            f.write("|-----------|-----------|---------|----------|----------|-------|\n")
            
            for algo, data in metrics.items():
                avg_survivors = np.mean(data["survivors_found"])
                avg_reward = np.mean(data["episode_rewards"])
                avg_battery = np.mean(data["battery_efficiency"])
                avg_time = np.mean(data["completion_time"])
                avg_steps = np.mean(data["steps_taken"])
                
                f.write(f"| {algo} | {avg_survivors:.2f} | {avg_reward:.2f} | "
                       f"{avg_battery:.2f}% | {avg_time:.2f} | {avg_steps:.1f} |\n")
            
            # Detailed metrics
            for algo, data in metrics.items():
                f.write(f"\n## {algo} Detailed Metrics\n\n")
                f.write(f"- Average Survivors Found: {np.mean(data['survivors_found']):.2f} "
                       f"(±{np.std(data['survivors_found']):.2f})\n")
                f.write(f"- Average Episode Reward: {np.mean(data['episode_rewards']):.2f} "
                       f"(±{np.std(data['episode_rewards']):.2f})\n")
                f.write(f"- Average Battery Remaining: {np.mean(data['battery_efficiency']):.2f}% "
                       f"(±{np.std(data['battery_efficiency']):.2f}%)\n")
                f.write(f"- Average Completion Time: {np.mean(data['completion_time']):.2f}s "
                       f"(±{np.std(data['completion_time']):.2f}s)\n")
                f.write(f"- Average Steps per Episode: {np.mean(data['steps_taken']):.1f} "
                       f"(±{np.std(data['steps_taken']):.1f})\n")

    def _plot_performance_comparison(self, metrics):
        """Generate performance comparison plots with environment-matching colors"""
        # Colors from environment
        env_colors = {
            "DQN": (30/255, 60/255, 200/255),
            "PPO": (0/255, 200/255, 0/255),
            "A2C": (220/255, 40/255, 40/255),
            "REINFORCE": (255/255, 220/255, 0/255)
        }
        metrics_to_plot = {
            "Survivors Found": "survivors_found",
            "Episode Rewards": "episode_rewards",
            "Battery Efficiency": "battery_efficiency",
            "Steps Taken": "steps_taken"
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Algorithm Performance Comparison", fontsize=16)
        
        for (title, metric), ax in zip(metrics_to_plot.items(), axes.flat):
            data = [metrics[algo][metric] for algo in metrics.keys()]
            box = ax.boxplot(data, patch_artist=True, labels=list(metrics.keys()))
            for patch, algo in zip(box['boxes'], metrics.keys()):
                patch.set_facecolor(env_colors[algo])
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, "performance_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Performance plots saved to: {save_path}")

def main():
    """Main function to run evaluations and recordings"""
    print("🚁 Drone Rescue Algorithm Evaluation and Recording")
    print("=" * 50)
    
    evaluator = DroneRescueEvaluator()
    
    # Record videos for each algorithm
    print("\n📹 Recording Algorithm Performances...")
    for algo in ["DQN", "PPO", "A2C", "REINFORCE"]:
        evaluator.record_algorithm(algo)
    
    # Run performance analysis
    print("\n📊 Running Performance Analysis...")
    metrics = evaluator.analyze_performance()
    
    print("\n✅ Evaluation complete! Check results directory for detailed analysis.")

if __name__ == "__main__":
    main()