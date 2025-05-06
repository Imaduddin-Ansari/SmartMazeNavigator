import pygame
import numpy as np
import random
import heapq
from collections import deque
import time

# Initialize pygame
pygame.init()

# Constants
CELL_SIZE = 20
MAZE_WIDTH = 41  
MAZE_HEIGHT = 41 

# Add margins to create space at top and bottom
TOP_MARGIN = 50
BOTTOM_MARGIN = 50

SCREEN_WIDTH = MAZE_WIDTH * CELL_SIZE
SCREEN_HEIGHT = MAZE_HEIGHT * CELL_SIZE + TOP_MARGIN + BOTTOM_MARGIN

BG_COLOR = (0, 0, 0)
WALL_COLOR = (50, 50, 50)
PATH_COLOR = (200, 200, 255)
PLAYER_COLOR = (255, 0, 0)
START_COLOR = (0, 255, 0)
END_COLOR = (255, 255, 0)
ASTAR_PATH_COLOR = (255, 100, 100)
VISITED_COLOR = (100, 100, 255, 128)

# Main menu colors - You can modify these colors
TITLE_COLOR = (255, 255, 255)  # Title text color - Change here
BUTTON_COLOR = (100, 100, 200)
BUTTON_HOVER_COLOR = (150, 150, 255)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Smart Maze Navigator")
clock = pygame.time.Clock()

# Additional colors for multiple paths
PATH_TYPES = [
    (200, 200, 255),  # Default path
    (200, 255, 200),  # Alternative path 1
    (255, 200, 200),  # Alternative path 2
    (255, 255, 200),  # Alternative path 3
]

class Button:
    def __init__(self, x, y, width, height, text, font_size=30):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.SysFont(None, font_size)  # You can change 'None' to a specific font name
        self.is_hovered = False
    
    def draw(self, surface):
        # Draw button with hover effect
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BUTTON_TEXT_COLOR, self.rect, 2)  # Border
        
        # Draw text
        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)  # Change BUTTON_TEXT_COLOR here
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered
    
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False

class MainMenu:
    def __init__(self):
        self.running = True
        self.play_game = False
        
        # Create buttons
        button_width, button_height = 200, 60
        button_x = SCREEN_WIDTH // 2 - button_width // 2
        
        self.play_button = Button(
            button_x, SCREEN_HEIGHT // 2, 
            button_width, button_height, "Play"
        )
        
        self.multiplayer_button = Button(
            button_x, SCREEN_HEIGHT // 2 + button_height + 20, 
            button_width, button_height, "Multiplayer"
        )
        
        # Title font - You can change font family, size, and style here
        self.title_font = pygame.font.SysFont("arial", 72)  # Change "arial" to preferred font

        try:
            # Replace 'background.jpg' with your image file path
            self.background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.background.fill((20, 20, 50))  # Fallback color - dark blue
            
            # Draw some decorative elements as placeholder
            for i in range(50):
                x = random.randint(0, SCREEN_WIDTH)
                y = random.randint(0, SCREEN_HEIGHT)
                radius = random.randint(1, 3)
                pygame.draw.circle(self.background, (255, 255, 255, 50), (x, y), radius)
        except:
            # Fallback if image loading fails
            self.background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.background.fill((20, 20, 50))  # Dark blue
    
    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        
        # Check if buttons are hovered
        self.play_button.check_hover(mouse_pos)
        self.multiplayer_button.check_hover(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Check for button clicks
            if self.play_button.is_clicked(mouse_pos, event):
                self.play_game = True
                self.running = False
            
            if self.multiplayer_button.is_clicked(mouse_pos, event):
                # For now, just display a message since multiplayer isn't implemented
                print("Multiplayer mode is not yet implemented!")
    
    def draw(self):
        # Use the background image instead of a solid color
        screen.blit(self.background, (0, 0))
        
        # Draw title with shadow for better visibility over background
        title_text = "Smart Maze Navigator"
        # Draw shadow first
        shadow_surf = self.title_font.render(title_text, True, (0, 0, 0))
        shadow_rect = shadow_surf.get_rect(center=(SCREEN_WIDTH // 2 + 2, SCREEN_HEIGHT // 3 + 2))
        screen.blit(shadow_surf, shadow_rect)
        
        # Draw actual title
        title_surf = self.title_font.render(title_text, True, TITLE_COLOR)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        screen.blit(title_surf, title_rect)
        
        # Add subtitle with different font/color
        subtitle_font = pygame.font.SysFont("arial", 28)  # Smaller font for subtitle
        subtitle_surf = subtitle_font.render("Navigate through challenging mazes!", True, (200, 200, 255))
        subtitle_rect = subtitle_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3 + 50))
        screen.blit(subtitle_surf, subtitle_rect)
        
        # Draw buttons
        self.play_button.draw(screen)
        self.multiplayer_button.draw(screen)
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.draw()
            clock.tick(60)
        
        return self.play_game


class MazeGenerator:
    def __init__(self, width, height):
        # Ensure dimensions are odd
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.maze = np.ones((self.height, self.width), dtype=int)
    
    def generate_maze(self):
        # Reset maze to all walls
        self.maze = np.ones((self.height, self.width), dtype=int)
        
        # Start with a grid full of walls
        for i in range(self.height):
            for j in range(self.width):
                self.maze[i][j] = 1
        
        # Create a starting point
        start_x, start_y = 1, 1
        self.maze[start_y][start_x] = 0
        
        # Use DFS to carve passages
        stack = [(start_x, start_y)]
        visited = set([(start_x, start_y)])
        
        while stack:
            x, y = stack[-1]
            # Find all neighboring cells not visited
            neighbors = []
            # Check in all four directions (up, right, down, left)
            for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.width-1 and 0 < ny < self.height-1 and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Choose one neighbor at random
                nx, ny = random.choice(neighbors)
                # Remove the wall between the current cell and the chosen cell
                self.maze[y + (ny - y) // 2][x + (nx - x) // 2] = 0
                # Mark the chosen cell as visited
                self.maze[ny][nx] = 0
                # Add the chosen cell to the stack
                stack.append((nx, ny))
                visited.add((nx, ny))
            else:
                # Backtrack
                stack.pop()
        
        # Add additional paths to create loops and alternative routes
        self._add_additional_paths()
        
        # Verify and fix any path issues
        self._verify_and_fix_paths()
        
        # Set start and end positions
        self.start = (1, 1)
        self.end = (self.width - 2, self.height - 2)
        
        # Ensure start and end are open
        self.maze[self.start[1]][self.start[0]] = 0
        self.maze[self.end[1]][self.end[0]] = 0
        
        # Ensure there's a valid path from start to end
        self._ensure_valid_path(self.start, self.end)
        
        return self.maze, self.start, self.end
        
    def _add_additional_paths(self):
        # Add some random connections to create loops and alternative paths
        min_paths = self.width // 4
        max_paths = int(self.width // 2)  # Convert to integer explicitly
        num_additional_paths = random.randint(min_paths, max_paths)
        
        # Add random connections
        for _ in range(num_additional_paths):
            # Pick a random wall that's not on the edge
            x, y = 0, 0
            attempts = 0
            while (self.maze[y][x] == 0 or x == 0 or y == 0 or 
                  x == self.width-1 or y == self.height-1) and attempts < 100:
                x = random.randint(1, self.width-2)
                y = random.randint(1, self.height-2)
                attempts += 1
            
            if attempts >= 100:
                continue  # Skip if we couldn't find a suitable wall
            
            # Check if breaking this wall creates a valid path (connects two open spaces)
            valid_directions = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx1, ny1 = x + dx, y + dy
                nx2, ny2 = x - dx, y - dy
                
                if (0 <= nx1 < self.width and 0 <= ny1 < self.height and
                    0 <= nx2 < self.width and 0 <= ny2 < self.height):
                    if self.maze[ny1][nx1] == 0 and self.maze[ny2][nx2] == 0:
                        valid_directions.append((dx, dy))
            
            # If we found valid directions to create a path, pick one randomly
            if valid_directions:
                dx, dy = random.choice(valid_directions)
                self.maze[y][x] = 0  # Break the wall
    
    def _verify_and_fix_paths(self):
        """Scan the maze for isolated paths and fix them"""
        # Fix isolated cells by ensuring every open cell is connected to another open cell
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if self.maze[y][x] == 0:
                    # Count connections to this cell
                    connections = 0
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny][nx] == 0:
                            connections += 1
                    
                    # If this is an isolated cell, connect it to a neighbor
                    if connections == 0:
                        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        random.shuffle(directions)
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                self.maze[ny][nx] = 0
                                break
    
    def _ensure_valid_path(self, start, end):
        """Ensure there's a valid path from start to end using BFS"""
        # If there's no path, create one
        if not self._has_path(start, end):
            self._create_path(start, end)
    
    def _has_path(self, start, end):
        """Check if there's a path from start to end using BFS"""
        queue = deque([start])
        visited = set([start])
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == end:
                return True
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.maze[ny][nx] == 0 and (nx, ny) not in visited):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        
        return False
    
    def _create_path(self, start, end):
        """Create a direct path from start to end"""
        x, y = start
        ex, ey = end
        
        # First move horizontally
        while x != ex:
            x += 1 if x < ex else -1
            self.maze[y][x] = 0
        
        # Then move vertically
        while y != ey:
            y += 1 if y < ey else -1
            self.maze[y][x] = 0

class AStar:
    def __init__(self, maze):
        self.maze = maze
        self.height, self.width = maze.shape
    
    def update_maze(self, maze):
        """Update the maze after it has been changed"""
        self.maze = maze
        self.height, self.width = maze.shape
    
    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors
    
    def find_path(self, start, end):
        # Initialize data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}
        open_set_hash = {start}
        visited = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            visited.add(current)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path, visited
            
            for neighbor in self.get_neighbors(current):
                # Tentative g_score
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, end)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        return [], visited

class Game:
    def __init__(self):
        self.maze_generator = MazeGenerator(MAZE_WIDTH, MAZE_HEIGHT)
        self.reset_game()
        self.running = True
        self.return_to_menu = False
    
    def reset_game(self):
        """Reset or initialize the game state"""
        self.maze, self.start, self.end = self.maze_generator.generate_maze()
        self.player_pos = self.start
        self.astar = AStar(self.maze)
        self.astar_path, self.visited_cells = self.astar.find_path(self.start, self.end)
        self.success = False  # Reset success flag
        self.show_astar_path = True
        self.show_visited = True
        self.steps_taken = 0
        self.optimal_steps = len(self.astar_path) - 1 if self.astar_path else 0
        self.timer_start = time.time()
        self.elapsed_time = 0
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                # Movement with WASD keys
                x, y = self.player_pos
                new_x, new_y = x, y
                
                if event.key == pygame.K_w:
                    new_y = y - 1
                elif event.key == pygame.K_s:
                    new_y = y + 1
                elif event.key == pygame.K_a:
                    new_x = x - 1
                elif event.key == pygame.K_d:
                    new_x = x + 1
                elif event.key == pygame.K_r:
                    # Reset game
                    self.reset_game()
                    return
                elif event.key == pygame.K_p:
                    # Toggle A* path visibility
                    self.show_astar_path = not self.show_astar_path
                elif event.key == pygame.K_v:
                    # Toggle visited cells visibility
                    self.show_visited = not self.show_visited
                elif event.key == pygame.K_n:
                    # Generate a new maze with same dimensions but different layout
                    self.reset_game()
                    return
                elif event.key == pygame.K_ESCAPE:
                    # Return to main menu
                    self.return_to_menu = True
                    self.running = False
                
                # Check if the new position is valid (not a wall)
                if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and self.maze[new_y][new_x] == 0:
                    old_pos = self.player_pos
                    self.player_pos = (new_x, new_y)
                    # Only count a step if actually moving to a new position
                    if old_pos != self.player_pos:
                        self.steps_taken += 1
                
                # Check if player reached the end
                if self.player_pos == self.end and not self.success:
                    self.success = True
                    # Stop the timer when succeeding
                    self.elapsed_time = time.time() - self.timer_start
    
    def draw(self):
        screen.fill(BG_COLOR)
        
        # Draw status info at the top margin
        self.draw_status_info()
        
        # Draw the maze with offset for top margin
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
                if self.maze[y][x] == 1:  # Wall
                    pygame.draw.rect(screen, WALL_COLOR, rect)
                else:  # Path
                    pygame.draw.rect(screen, PATH_COLOR, rect)
        
        # Draw visited cells (A* algorithm)
        if self.show_astar_path and self.show_visited:
            for x, y in self.visited_cells:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, VISITED_COLOR, rect)
        
        # Draw A* path
        if self.show_astar_path:
            for x, y in self.astar_path:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, ASTAR_PATH_COLOR, rect)
        
        # Draw start and end positions
        start_rect = pygame.Rect(self.start[0] * CELL_SIZE, self.start[1] * CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
        end_rect = pygame.Rect(self.end[0] * CELL_SIZE, self.end[1] * CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, START_COLOR, start_rect)
        pygame.draw.rect(screen, END_COLOR, end_rect)
        
        # Draw player
        player_rect = pygame.Rect(self.player_pos[0] * CELL_SIZE, self.player_pos[1] * CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, PLAYER_COLOR, player_rect)
        
        # Update time if game is active and not successful yet
        if not self.success:
            self.elapsed_time = time.time() - self.timer_start
        
        # Draw controls info in the bottom margin
        self.draw_controls_info()
        
        # Draw success message if player reached the end
        if self.success:
            self.draw_success_popup()
        
        pygame.display.flip()
    
    def draw_status_info(self):
        """Draw game status information in the top margin"""
        font = pygame.font.SysFont(None, 22)
        
        # Draw steps counter
        steps_text = font.render(f"Steps: {self.steps_taken} (Optimal: {self.optimal_steps})", 
                                True, (255, 255, 255))
        screen.blit(steps_text, (10, 10))
        
        # Draw timer
        time_text = font.render(f"Time: {self.elapsed_time:.1f}s", True, (255, 255, 255))
        screen.blit(time_text, (10, 30))
        
        # Draw legend
        legend_start = font.render("Green: Start", True, START_COLOR)
        screen.blit(legend_start, (SCREEN_WIDTH - 200, 10))
        
        legend_end = font.render("Yellow: End", True, END_COLOR)
        screen.blit(legend_end, (SCREEN_WIDTH - 200, 30))
    
    def draw_controls_info(self):
        """Draw controls information in the bottom margin"""
        font = pygame.font.SysFont(None, 18)
        controls_text = font.render("Controls: WASD to move, R to reset, P to toggle A* path, V to toggle visited cells, ESC for menu", 
                                   True, (255, 255, 255))
        screen.blit(controls_text, (10, SCREEN_HEIGHT - 30))
    
    def draw_success_popup(self):
        """Draw success popup with game statistics"""
        # Calculate efficiency
        efficiency = self.optimal_steps / max(1, self.steps_taken) * 100
        
        # Background for success message
        bg_rect = pygame.Rect(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 3, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3)
        pygame.draw.rect(screen, (0, 0, 100), bg_rect)
        pygame.draw.rect(screen, (255, 255, 255), bg_rect, 2)
        
        # Success message
        font = pygame.font.SysFont(None, 36)
        text = font.render("Success!", True, (255, 255, 255))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40))
        screen.blit(text, text_rect)
        
        # Stats
        font = pygame.font.SysFont(None, 24)
        time_text = font.render(f"Time: {self.elapsed_time:.1f}s", True, (255, 255, 255))
        time_rect = time_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(time_text, time_rect)
        
        steps_text = font.render(f"Your steps: {self.steps_taken}", True, (255, 255, 255))
        steps_rect = steps_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 25))
        screen.blit(steps_text, steps_rect)
        
        optimal_text = font.render(f"Optimal path: {self.optimal_steps} steps", True, (255, 255, 255))
        optimal_rect = optimal_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        screen.blit(optimal_text, optimal_rect)
        
        efficiency_text = font.render(f"Efficiency: {efficiency:.1f}%", True, (255, 255, 255))
        efficiency_rect = efficiency_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 75))
        screen.blit(efficiency_text, efficiency_rect)
        
        restart_text = font.render("Press 'R' for new maze or 'N' for new layout", True, (255, 255, 255))
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 110))
        screen.blit(restart_text, restart_rect)
    
    def run(self):
        while self.running:
            self.handle_events()
            self.draw()
            clock.tick(60)
        
        return self.return_to_menu

# Main function
def main():
    running = True
    
    while running:
        # Start with the main menu
        menu = MainMenu()
        play_game = menu.run()
        
        if play_game:
            # If "Play" was selected, start the game
            game = Game()
            return_to_menu = game.run()
            
            # If ESC was pressed, we'll return to the menu
            # Otherwise, if the window was closed, we'll exit
            if not return_to_menu:
                running = False
        else:
            # If the menu was closed, exit
            running = False
    
    pygame.quit()

if __name__ == "__main__":
    main()