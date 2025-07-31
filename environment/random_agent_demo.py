import sys
import os
import time
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import EmergencyDroneEnv

def find_path(start, goal):
    """Simple BFS for shortest path in grid."""
    from collections import deque
    queue = deque()
    queue.append((start, []))
    visited = set()
    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path
        if pos in visited:
            continue
        visited.add(pos)
        x, y = pos
        for dx, dy, action in [ (0,-1,0), (0,1,1), (-1,0,2), (1,0,3) ]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < 10 and 0 <= ny < 10:
                queue.append( ((nx,ny), path+[action]) )
    return []

def run_rescue_all_agent(steps=100, delay=0.1, save_gif=True, gif_path=None):
    env = EmergencyDroneEnv(render_mode="rgb_array")
    obs, info = env.reset()
    frames = []

    # Set default gif_path to assets directory in project root
    if gif_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(project_root, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        gif_path = os.path.join(assets_dir, "rescue_all_agent_demo.gif")
    else:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    survivors_to_find = set(tuple(pos) for pos in env.survivors)
    found = set()
    drone_pos = tuple(env.drone_pos)
    actions = []
    step = 0

    # Plan: For each survivor, go to them, scan, then go to next
    for survivor in survivors_to_find:
        # Move to survivor
        path = find_path(drone_pos, survivor)
        actions.extend(path)
        actions.append(4)  # Scan
        drone_pos = survivor
    # After all, return to base
    path = find_path(drone_pos, env.base_pos)
    actions.extend(path)
    actions.append(5)  # Return to base

    # Pad with random actions if needed
    while len(actions) < steps:
        actions.append(env.action_space.sample())

    for i in range(steps):
        action = actions[i]
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if save_gif:
            frames.append(frame)
        print(f"Step {i+1}: Action={action}, Reward={reward}, Obs={obs}")
        time.sleep(delay)
        if terminated or truncated:
            print("Episode ended early.")
            break

    print("Demo finished. Closing environment in 30 seconds...")
    time.sleep(60)
    env.close()

    if save_gif and frames:
        imageio.mimsave(gif_path, frames, duration=delay)
        print(f"GIF saved to {gif_path}")

if __name__ == "__main__":
    run_rescue_all_agent(steps=200, delay=0.1, save_gif=True)