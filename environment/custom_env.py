import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class EmergencyDroneEnv(gym.Env):
    """
    Emergency Response Drone Environment

    Mission: Navigate disaster area to locate survivors while managing battery

    Observation Space:
    - Drone position (x, y)
    - Battery level (0-100)
    - Survivors found / total survivors
    - Steps taken

    Action Space:
    0: Move Up
    1: Move Down  
    2: Move Left
    3: Move Right
    4: Scan Area
    5: Return to Base
    6: Emergency Land
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=10):
        super().__init__()

        # Environment parameters
        self.grid_size = grid_size
        self.render_mode = render_mode

        # Action space: 7 discrete actions
        self.action_space = spaces.Discrete(7)

        # Observation space: [drone_x, drone_y, battery, survivors_found, total_survivors, steps_taken]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([grid_size-1, grid_size-1, 100, 10, 10, 1000]),
            dtype=np.float32
        )

        # Initialize pygame for rendering
        self.window_size = 600
        self.window = None
        self.clock = None

        # Environment state
        self.drone_pos = None
        self.base_pos = (0, 0)  # Base station at origin
        self.battery = 100
        self.survivors = []  # List of survivor positions
        self.found_survivors = set()  # Set of found survivor positions
        self.scanned_areas = set()  # Areas that have been scanned
        self.steps_taken = 0
        self.max_steps = 500
        self.battery_critical = 30  # New: Critical battery threshold
        self.movement_reward = -0.1  # New: Base movement cost

        # Mission parameters
        self.num_survivors = random.randint(10, 15)  # Random number of survivors
        self.scan_radius = 2  # How far drone can scan
        self.battery_drain_move = 2  # Battery cost for moving
        self.battery_drain_scan = 5  # Battery cost for scanning
        self.battery_recharge_rate = 20  # Battery gained at base

    def _generate_survivors(self):
        """Generate random survivor locations avoiding base"""
        self.survivors = []
        self.found_survivors = set()

        for _ in range(self.num_survivors):
            while True:
                pos = (random.randint(1, self.grid_size-1), 
                       random.randint(1, self.grid_size-1))
                if pos != self.base_pos and pos not in self.survivors:
                    self.survivors.append(pos)
                    break

    def _get_observation(self):
        """Return current observation"""
        return np.array([
            self.drone_pos[0],
            self.drone_pos[1], 
            self.battery,
            len(self.found_survivors),
            len(self.survivors),
            self.steps_taken
        ], dtype=np.float32)

    def _calculate_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_in_scan_range(self, target_pos):
        """Check if target position is within scan range"""
        return self._calculate_distance(self.drone_pos, target_pos) <= self.scan_radius

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Reset drone to base
        self.drone_pos = list(self.base_pos)
        self.battery = 100
        self.steps_taken = 0
        self.scanned_areas = set()

        # Generate new survivors
        self._generate_survivors()

        return self._get_observation(), {}

    def step(self, action):
        """Execute action and return results"""
        self.steps_taken += 1
        reward = -0.1  # Smaller base penalty
        terminated = False
        truncated = False
        info = {}

        # Store previous state
        previous_pos = tuple(self.drone_pos)
        previous_dist = min([self._calculate_distance(self.drone_pos, surv) 
                           for surv in self.survivors if surv not in self.found_survivors],
                          default=self.grid_size * 2)

        # Execute action
        if action == 0:  # Move Up
            if self.drone_pos[1] > 0:
                self.drone_pos[1] -= 1
                self.battery -= self.battery_drain_move
                reward += self.movement_reward

        elif action == 1:  # Move Down
            if self.drone_pos[1] < self.grid_size - 1:
                self.drone_pos[1] += 1
                self.battery -= self.battery_drain_move
                reward += self.movement_reward

        elif action == 2:  # Move Left
            if self.drone_pos[0] > 0:
                self.drone_pos[0] -= 1
                self.battery -= self.battery_drain_move
                reward += self.movement_reward

        elif action == 3:  # Move Right
            if self.drone_pos[0] < self.grid_size - 1:
                self.drone_pos[0] += 1
                self.battery -= self.battery_drain_move
                reward += self.movement_reward

        # Add reward for getting closer to survivors after movement
        if action in [0, 1, 2, 3]:  # Movement actions
            current_dist = min([self._calculate_distance(self.drone_pos, surv) 
                              for surv in self.survivors if surv not in self.found_survivors],
                             default=self.grid_size * 2)
            if current_dist < previous_dist:
                reward += 2  # Reward for getting closer to survivors
            elif current_dist > previous_dist:
                reward -= 0.5  # Small penalty for getting further

        elif action == 4:  # Scan Area
            self.battery -= self.battery_drain_scan
            scan_pos = tuple(self.drone_pos)
            new_scan = scan_pos not in self.scanned_areas
            self.scanned_areas.add(scan_pos)

            # Check for survivors in scan range
            survivors_found_this_scan = 0
            for survivor_pos in self.survivors:
                if (survivor_pos not in self.found_survivors and 
                    self._is_in_scan_range(survivor_pos)):
                    self.found_survivors.add(survivor_pos)
                    survivors_found_this_scan += 1
                    reward += 100  # Big reward for finding survivor

            # Reward for scanning new areas
            if new_scan:
                reward += 5
            
            # Add scan efficiency bonus
            if survivors_found_this_scan > 0:
                reward += 10 * survivors_found_this_scan

            info['survivors_found'] = survivors_found_this_scan

        elif action == 5:  # Return to Base
            if tuple(self.drone_pos) == self.base_pos:
                old_battery = self.battery
                self.battery = min(100, self.battery + self.battery_recharge_rate)
                reward += (self.battery - old_battery) * 0.2  # Proportional to battery gained
            else:
                # Guide the agent back to base when battery is low
                if self.battery < self.battery_critical:
                    dist_to_base = self._calculate_distance(self.drone_pos, self.base_pos)
                    reward -= dist_to_base * 0.5

        elif action == 6:  # Emergency Land
            if self.battery < 20:  # Smart emergency landing
                reward += 10
            else:
                reward -= 20
            terminated = True
            info['termination_reason'] = 'emergency_land'

        # Check termination conditions
        if self.battery <= 0:
            reward -= 50
            terminated = True
            info['termination_reason'] = 'battery_depleted'

        if len(self.found_survivors) == len(self.survivors):
            reward += 200
            terminated = True
            info['termination_reason'] = 'mission_complete'

        if self.steps_taken >= self.max_steps:
            truncated = True
            info['termination_reason'] = 'max_steps'

        # Calculate exploration bonus with diminishing returns
        exploration_bonus = len(self.scanned_areas) * 0.5 / max(1, self.steps_taken/100)
        reward += exploration_bonus

        # Add detailed info
        info.update({
            'battery': self.battery,
            'survivors_found': len(self.found_survivors),
            'total_survivors': len(self.survivors),
            'steps': self.steps_taken,
            'scanned_areas': len(self.scanned_areas),
            'position': tuple(self.drone_pos),
            'base_distance': self._calculate_distance(self.drone_pos, self.base_pos)
        })

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        """Render current frame using pygame"""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Emergency Drone - Festus Bigirimana")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (0, 0, 255)      # Drone
        GREEN = (0, 255, 0)     # Base
        RED = (255, 0, 0)       # Survivors
        YELLOW = (255, 255, 0)  # Found survivors
        GRAY = (128, 128, 128)  # Scanned areas

        # Clear screen
        self.window.fill(WHITE)

        # Calculate cell size
        cell_size = self.window_size // self.grid_size

        # Draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(self.window, BLACK, 
                           (x * cell_size, 0), 
                           (x * cell_size, self.window_size))
        for y in range(self.grid_size + 1):
            pygame.draw.line(self.window, BLACK, 
                           (0, y * cell_size), 
                           (self.window_size, y * cell_size))

        # Draw scanned areas
        for pos in self.scanned_areas:
            rect = pygame.Rect(pos[0] * cell_size, pos[1] * cell_size, 
                             cell_size, cell_size)
            pygame.draw.rect(self.window, GRAY, rect)

        # Draw base
        base_rect = pygame.Rect(self.base_pos[0] * cell_size, 
                               self.base_pos[1] * cell_size, 
                               cell_size, cell_size)
        pygame.draw.rect(self.window, GREEN, base_rect)

        # Draw survivors
        for survivor_pos in self.survivors:
            color = YELLOW if survivor_pos in self.found_survivors else RED
            center = (survivor_pos[0] * cell_size + cell_size // 2,
                     survivor_pos[1] * cell_size + cell_size // 2)
            pygame.draw.circle(self.window, color, center, cell_size // 3)

        # Draw drone
        drone_center = (self.drone_pos[0] * cell_size + cell_size // 2,
                       self.drone_pos[1] * cell_size + cell_size // 2)
        pygame.draw.circle(self.window, BLUE, drone_center, cell_size // 2)

        # Draw battery indicator
        battery_width = int((self.battery / 100) * 100)
        battery_rect = pygame.Rect(10, 10, battery_width, 20)
        pygame.draw.rect(self.window, GREEN if self.battery > 30 else RED, battery_rect)

        # Add text labels
        font = pygame.font.Font(None, 24)
        battery_text = font.render(f"Battery: {self.battery:.1f}%", True, BLACK)
        self.window.blit(battery_text, (10, 35))

        survivors_text = font.render(f"Survivors: {len(self.found_survivors)}/{len(self.survivors)}", True, BLACK)
        self.window.blit(survivors_text, (10, 55))

        steps_text = font.render(f"Steps: {self.steps_taken}", True, BLACK)
        self.window.blit(steps_text, (10, 75))

        # Update display
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )

    def close(self):
        """Clean up pygame"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()