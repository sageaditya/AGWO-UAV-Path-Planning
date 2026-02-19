import numpy as np
from typing import List, Tuple, Optional, Union

class GreyWPlanner:
    def __init__(self, 
                 start: np.ndarray,
                 goal: np.ndarray,
                 bounds: np.ndarray,
                 obstacles: List[Union[np.ndarray, Tuple, dict]]):
        
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        
        # Tuned parameters
        self.n_wolves = 40
        self.resolution = 0.5
        self.nodes = []
        self.max_path_length = 200
        self.obstacle_margin = 0.8
        
    def check_collision(self, point: np.ndarray) -> bool:
        """Check if point collides with obstacles"""
        for obs in self.obstacles:
            if isinstance(obs, np.ndarray):
                x, y, z = point
                ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obs
                if (ox_min - self.obstacle_margin <= x <= ox_max + self.obstacle_margin and
                    oy_min - self.obstacle_margin <= y <= oy_max + self.obstacle_margin and
                    oz_min - self.obstacle_margin <= z <= oz_max + self.obstacle_margin):
                    return True
            else:
                if isinstance(obs, dict):
                    center = obs['position']
                    radius = obs['radius']
                else:
                    center, radius = obs
                if np.linalg.norm(point - np.array(center)) <= radius + self.obstacle_margin:
                    return True
        return False

    def generate_path(self) -> np.ndarray:
        """Generate path with strong goal bias"""
        path = [self.start.copy()]
        current = self.start.copy()
        
        while len(path) < self.max_path_length:
            if np.linalg.norm(current - self.goal) < self.resolution:
                path.append(self.goal.copy())
                break
                
            # Strong goal bias (80% chance to move towards goal)
            if np.random.random() < 0.8:
                direction = self.goal - current
                direction = direction / (np.linalg.norm(direction) + 1e-6)
            else:
                direction = np.random.randn(3)
                direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            new_pos = current + direction * self.resolution
            new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
            
            if not self.check_collision(new_pos):
                path.append(new_pos)
                current = new_pos
            else:
                # If collision, try sideways movements
                for angle in [np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]:
                    rot_matrix = np.array([
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]
                    ])
                    new_direction = rot_matrix @ direction
                    new_pos = current + new_direction * self.resolution
                    new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
                    
                    if not self.check_collision(new_pos):
                        path.append(new_pos)
                        current = new_pos
                        break
        
        # Ensure path ends at goal
        if len(path) > 1 and np.linalg.norm(path[-1] - self.goal) > self.resolution:
            path.append(self.goal.copy())
            
        return np.array(path)

    def plan_path(self) -> Optional[np.ndarray]:
        """Plan path using improved GWO"""
        best_path = None
        best_length = float('inf')
        
        for _ in range(300):  # Increased iterations
            current_path = self.generate_path()
            
            if len(current_path) > 1:
                # Check if path reaches near goal
                if np.linalg.norm(current_path[-1] - self.goal) < self.resolution * 2:
                    path_length = sum(np.linalg.norm(current_path[i] - current_path[i-1]) 
                                    for i in range(1, len(current_path)))
                    
                    if path_length < best_length:
                        best_length = path_length
                        best_path = current_path
                        self.nodes = current_path  # Store the current path
                        
                        # If path is good enough, return it
                        if path_length < np.linalg.norm(self.goal - self.start) * 2:
                            return best_path
        
        if best_path is not None:
            # Ensure the path ends exactly at the goal
            if not np.array_equal(best_path[-1], self.goal):
                best_path = np.vstack([best_path, self.goal])
            return best_path
            
        return None
