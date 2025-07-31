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
        self.battery_critical = 30  # Critical battery threshold
        self.movement_reward = -0.1  # Base movement cost

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
        """Render current frame using pygame with advanced fake 3D (isometric) visualization"""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Emergency Drone - Festus Bigirimana (Fake 3D)")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Colors
        SKY = (135, 206, 235)
        GROUND = (120, 180, 90)
        GRID = (80, 120, 60)
        SHADOW = (60, 60, 60, 80)
        DRONE_BODY = (60, 60, 60)
        DRONE_ARM = (120, 120, 120)
        PROPELLER = (180, 180, 180)
        DRONE_HIGHLIGHT = (30, 60, 200)
        BASE = (0, 200, 0)
        BASE_TOP = (0, 255, 100)
        RED = (220, 40, 40)
        YELLOW = (255, 220, 0)
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GRAY = (128, 128, 128, 80)

        # Helper for isometric projection
        def iso(x, y, z=0):
            scale = self.window_size // (self.grid_size + 2)
            iso_x = int((x - y) * scale * 0.7 + self.window_size // 2)
            iso_y = int((x + y) * scale * 0.35 - z * scale * 0.7 + self.window_size // 4)
            return (iso_x, iso_y)

        # Fill sky
        self.window.fill(SKY)

        # Draw ground tiles (isometric)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pts = [
                    iso(x, y),
                    iso(x+1, y),
                    iso(x+1, y+1),
                    iso(x, y+1)
                ]
                pygame.draw.polygon(self.window, GROUND, pts)
                pygame.draw.polygon(self.window, GRID, pts, 1)

        # Draw scanned areas (semi-transparent overlay)
        scan_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        for pos in self.scanned_areas:
            x, y = pos
            pts = [
                iso(x, y),
                iso(x+1, y),
                iso(x+1, y+1),
                iso(x, y+1)
            ]
            pygame.draw.polygon(scan_surface, GRAY, pts)
        self.window.blit(scan_surface, (0, 0))

        # Draw base as a raised isometric cylinder
        bx, by = self.base_pos
        base_pts = [
            iso(bx, by, 0.2),
            iso(bx+1, by, 0.2),
            iso(bx+1, by+1, 0.2),
            iso(bx, by+1, 0.2)
        ]
        pygame.draw.polygon(self.window, BASE_TOP, base_pts)
        pygame.draw.polygon(self.window, BASE, [iso(bx, by), iso(bx+1, by), iso(bx+1, by+1), iso(bx, by+1)])
        # Draw H
        font = pygame.font.Font(None, 24)
        h_text = font.render("H", True, BLACK)
        h_pos = iso(bx+0.5, by+0.5, 0.25)
        self.window.blit(h_text, (h_pos[0]-8, h_pos[1]-12))

        # Draw survivors as isometric stick figures with shadows
        for survivor_pos in self.survivors:
            color = YELLOW if survivor_pos in self.found_survivors else RED
            sx, sy = survivor_pos
            center = iso(sx+0.5, sy+0.5, 0.15)
            # Draw survivor shadow
            survivor_shadow = pygame.Surface((28, 12), pygame.SRCALPHA)
            pygame.draw.ellipse(survivor_shadow, SHADOW, (0, 0, 28, 12))
            self.window.blit(survivor_shadow, (center[0]-14, center[1]+28))
            # Head
            pygame.draw.circle(self.window, color, center, 10)
            # Body
            pygame.draw.line(self.window, color, (center[0], center[1]+10), (center[0], center[1]+28), 3)
            # Arms
            pygame.draw.line(self.window, color, (center[0]-8, center[1]+18), (center[0]+8, center[1]+18), 3)
            # Legs
            pygame.draw.line(self.window, color, (center[0], center[1]+28), (center[0]-7, center[1]+38), 3)
            pygame.draw.line(self.window, color, (center[0], center[1]+28), (center[0]+7, center[1]+38), 3)

        # Draw drone shadow
        dx, dy = self.drone_pos
        drone_center = iso(dx+0.5, dy+0.5, 0.18)
        shadow_surface = pygame.Surface((40, 20), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, SHADOW, (0, 0, 40, 20))
        self.window.blit(shadow_surface, (drone_center[0]-20, drone_center[1]+18))

        # Draw drone as isometric quadcopter
        # Body
        pygame.draw.ellipse(self.window, DRONE_BODY, (drone_center[0]-12, drone_center[1]-10, 24, 20))
        # Arms and propellers
        for angle in [45, 135, 225, 315]:
            rad = np.deg2rad(angle)
            arm_end = (int(drone_center[0] + np.cos(rad) * 18), int(drone_center[1] + np.sin(rad) * 8))
            pygame.draw.line(self.window, DRONE_ARM, drone_center, arm_end, 4)
            pygame.draw.circle(self.window, PROPELLER, arm_end, 5)
        # Highlight
        pygame.draw.ellipse(self.window, DRONE_HIGHLIGHT, (drone_center[0]-6, drone_center[1]-5, 12, 10))

        # Draw scan radius (isometric, semi-transparent)
        scan_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        for r in range(1, self.scan_radius+1):
            for dx_scan in range(-r, r+1):
                for dy_scan in range(-r, r+1):
                    if abs(dx_scan) + abs(dy_scan) == r:
                        scan_tile = (dx+dx_scan, dy+dy_scan)
                        if 0 <= scan_tile[0] < self.grid_size and 0 <= scan_tile[1] < self.grid_size:
                            pts = [
                                iso(scan_tile[0], scan_tile[1], 0.01),
                                iso(scan_tile[0]+1, scan_tile[1], 0.01),
                                iso(scan_tile[0]+1, scan_tile[1]+1, 0.01),
                                iso(scan_tile[0], scan_tile[1]+1, 0.01)
                            ]
                            pygame.draw.polygon(scan_surface, (0, 0, 255, 30), pts)
        self.window.blit(scan_surface, (0, 0))

        # Draw battery indicator (3D style)
        battery_width = int((self.battery / 100) * 100)
        pygame.draw.rect(self.window, (40, 180, 40), (10, 10, battery_width, 20))
        pygame.draw.rect(self.window, BLACK, (10, 10, 100, 20), 2)
        pygame.draw.rect(self.window, (100, 100, 100), (110, 14, 8, 12))

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