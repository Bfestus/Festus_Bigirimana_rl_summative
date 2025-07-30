import sys
import os
import time
import imageio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import EmergencyDroneEnv

def run_random_agent(steps=100, delay=0.1, save_gif=True, gif_path="../assets/random_agent_demo.gif"):
    env = EmergencyDroneEnv(render_mode="rgb_array")
    obs, info = env.reset()
    frames = []

    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()  # Returns an RGB array
        if save_gif:
            frames.append(frame)
        print(f"Step {step+1}: Action={action}, Reward={reward}, Obs={obs}")
        time.sleep(delay)

    print("Demo finished. Closing environment in 3 seconds...")
    time.sleep(3)
    env.close()

    if save_gif:
        imageio.mimsave(gif_path, frames, duration=delay)
        print(f"GIF saved to {gif_path}")

if __name__ == "__main__":
    run_random_agent(steps=100, delay=0.1, save_gif=True, gif_path="../assets/random_agent_demo.gif")