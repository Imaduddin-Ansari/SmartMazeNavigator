import pygame
import numpy as np
import random
import heapq
from collections import deque
import time
from constraint import Problem, AllDifferentConstraint

# Initialize pygame
pygame.init()

# Constants
CELL_SIZE = 15
MAZE_WIDTH = 91  # Changed to make odd dimensions for CSP
MAZE_HEIGHT = 45  # Changed to make odd dimensions for CSP

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

# Main menu colors
TITLE_COLOR = (255, 255, 255)
BUTTON_COLOR = (100, 100, 200)
BUTTON_HOVER_COLOR = (150, 150, 255)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Smart Maze Navigator - CSP Edition")
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
        self.font = pygame.font.SysFont(None, font_size)
        self.is_hovered = False
    
    def draw(self, surface):
        # Draw button with hover effect
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BUTTON_TEXT_COLOR, self.rect, 2)  # Border
        
        # Draw text
        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
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
        
        # Title font
        self.title_font = pygame.font.SysFont("arial", 72)

        # Create background
        self.background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.background.fill((20, 20, 50))  # Dark blue
            
        # Draw some decorative elements as placeholder
        for i in range(50):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT)
            radius = random.randint(1, 3)
            pygame.draw.circle(self.background, (255, 255, 255, 50), (x, y), radius)
    
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
        subtitle_surf = subtitle_font.render("Navigate mazes generated with CSP!", True, (200, 200, 255))
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


class MazeCSP:
    """
    A class to generate mazes using Constraint Satisfaction Problems.
    
    Cell values:
    0 = Path (open)
    1 = Wall (blocked)
    """
    
    def __init__(self, width, height):
        """Initialize the maze generator with dimensions."""
        self.width = width
        self.height = height
        self.grid = None
        
    def setup_variables(self):
        """Set up the CSP variables - each cell in the grid is a variable."""
        # Create a new Problem instance each time to avoid constraints from previous runs
        self.problem = Problem()
        
        # Define variables for each cell in the grid
        for y in range(self.height):
            for x in range(self.width):
                # Each cell can be either a path (0) or wall (1)
                self.problem.addVariable((x, y), [0, 1])
                
    def add_border_constraint(self):
        """Add constraint to ensure the border cells are walls."""
        for y in range(self.height):
            for x in range(self.width):
                if x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1:
                    # Border cells must be walls
                    self.problem.addConstraint(lambda val: val == 1, [(x, y)])
        
    def add_entrance_exit_constraint(self, entrance_x=1, entrance_y=0, exit_x=None, exit_y=None):
        """Add constraints for entrance and exit."""
        if exit_x is None:
            exit_x = self.width - 2
        if exit_y is None:
            exit_y = self.height - 1
            
        # Entrance must be a path
        self.problem.addConstraint(lambda val: val == 0, [(entrance_x, entrance_y)])
        # Cell adjacent to entrance must be a path
        self.problem.addConstraint(lambda val: val == 0, [(entrance_x, entrance_y + 1)])
        
        # Exit must be a path
        self.problem.addConstraint(lambda val: val == 0, [(exit_x, exit_y)])
        # Cell adjacent to exit must be a path
        self.problem.addConstraint(lambda val: val == 0, [(exit_x, exit_y - 1)])
    
    def add_no_2x2_wall_constraint(self):
        """Add constraint to prevent 2x2 blocks of walls."""
        for y in range(self.height - 1):
            for x in range(self.width - 1):
                # Define the 2x2 grid cells
                top_left = (x, y)
                top_right = (x + 1, y)
                bottom_left = (x, y + 1)
                bottom_right = (x + 1, y + 1)
                
                # Add constraint: at least one of the 2x2 cells must be a path
                self.problem.addConstraint(
                    lambda tl, tr, bl, br: tl + tr + bl + br < 4,
                    [top_left, top_right, bottom_left, bottom_right]
                )
    
    def add_min_dead_ends_constraint(self, min_dead_ends=3):
        """
        Add constraint to ensure a minimum number of dead ends.
        Note: This is a complex constraint that is enforced post-solution.
        """
        # This will be handled in the solve method
        self.min_dead_ends = min_dead_ends
                
    def add_connectivity_constraint(self):
        """
        Add constraints to ensure maze is connected.
        This is a complex constraint that is enforced post-solution.
        """
        # This will be handled in the solve method
        pass
    
    def is_dead_end(self, x, y, grid):
        """Check if a cell is a dead end (path with only one neighboring path)."""
        if grid[y][x] != 0:  # Not a path
            return False
            
        # Count neighboring paths
        neighbors = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1)
        ]
        
        path_neighbors = 0
        for nx, ny in neighbors:
            if 0 <= nx < self.width and 0 <= ny < self.height and grid[ny][nx] == 0:
                path_neighbors += 1
                
        # A dead end has exactly one neighboring path
        return path_neighbors == 1
    
    def count_dead_ends(self, grid):
        """Count the number of dead ends in the maze."""
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.is_dead_end(x, y, grid):
                    count += 1
        return count
    
    def is_connected(self, grid, start=None, end=None):
        """Check if there's a path from start to end using BFS."""
        if start is None:
            # Find a path cell to start from
            for y in range(self.height):
                for x in range(self.width):
                    if grid[y][x] == 0:
                        start = (x, y)
                        break
                if start:
                    break
        
        if end is None:
            end = (self.width - 2, self.height - 2)
            
        # BFS to check connectivity
        visited = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            x, y = queue.pop(0)
            
            if (x, y) == end:
                return True
                
            # Check all four adjacent cells
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    grid[ny][nx] == 0 and (nx, ny) not in visited):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        
        # Check if all path cells are connected
        all_paths = sum(row.count(0) for row in grid)
        return len(visited) == all_paths
    
    def solve(self, max_attempts=50):
        """Solve the CSP and generate a valid maze."""
        # Set up the problem
        self.setup_variables()
        self.add_border_constraint()
        self.add_entrance_exit_constraint()
        self.add_no_2x2_wall_constraint()
        
        # Try to find a solution that meets all criteria
        for _ in range(max_attempts):
            # Get a solution
            solution = self.problem.getSolution()
            
            if not solution:
                return None
            
            # Convert solution to grid
            grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
            for (x, y), value in solution.items():
                grid[y][x] = value
            
            # Check connectivity
            if not self.is_connected(grid):
                continue
                
            # Check minimum dead ends
            if hasattr(self, 'min_dead_ends'):
                dead_ends = self.count_dead_ends(grid)
                if dead_ends < self.min_dead_ends:
                    continue
            
            # If we get here, we have a valid solution
            self.grid = grid
            return grid
            
        print("CSP failed to find a valid maze after max attempts.")
        return None
    
    def get_start_end(self):
        """Return start and end positions."""
        # Default start and end positions
        return (1, 1), (self.width - 2, self.height - 2)


def generate_maze_with_backtracking(width, height, min_dead_ends=3):
    """
    Alternative approach: Generate maze using recursive backtracking.
    This is more efficient for larger mazes where pure CSP would be slow.
    """
    # Initialize grid with all walls
    grid = np.ones((height, width), dtype=int)
    
    # Choose random starting point (must be odd coordinates)
    start_x = 1
    start_y = 1
    grid[start_y][start_x] = 0
    
    # Stack for backtracking
    stack = [(start_x, start_y)]
    
    # Directions: (dx, dy)
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    
    while stack:
        x, y = stack[-1]
        
        # Find unvisited neighbors
        unvisited = []
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 < nx < width-1 and 0 < ny < height-1 and grid[ny][nx] == 1):
                # Check if this cell has been visited
                if grid[ny][nx] == 1:
                    unvisited.append((nx, ny, dx//2, dy//2))
        
        if unvisited:
            # Choose random unvisited neighbor
            nx, ny, wall_x, wall_y = unvisited[0]
            
            # Remove wall
            grid[y + wall_y][x + wall_x] = 0
            grid[ny][nx] = 0
            
            # Push new cell to stack
            stack.append((nx, ny))
        else:
            # Backtrack
            stack.pop()
    
    # Add entrance and exit
    grid[0][1] = 0  # Entrance
    grid[height-1][width-2] = 0  # Exit
    
    # Ensure minimum dead ends
    dead_ends = count_dead_ends(grid, width, height)
    
    if dead_ends < min_dead_ends:
        # Add more dead ends if needed
        attempts = 0
        while dead_ends < min_dead_ends and attempts < 100:
            # Find a random junction and convert it to a dead end
            x, y = random.randint(1, width-2), random.randint(1, height-2)
            if grid[y][x] == 0 and not is_dead_end(x, y, grid, width, height):
                # Count path neighbors
                path_neighbors = []
                for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                    if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 0:
                        path_neighbors.append((nx, ny))
                
                if len(path_neighbors) >= 3:  # It's a junction
                    # Block all but one neighbor to create a dead end
                    for i in range(len(path_neighbors) - 1):
                        nx, ny = path_neighbors[i]
                        grid[ny][nx] = 1
                    
                    # Recount dead ends
                    dead_ends = count_dead_ends(grid, width, height)
            
            attempts += 1
    
    return grid

def is_dead_end(x, y, grid, width, height):
    """Check if a cell is a dead end in the given grid."""
    if grid[y][x] != 0:  # Not a path
        return False
        
    # Count neighboring paths
    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    path_neighbors = 0
    
    for nx, ny in neighbors:
        if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 0:
            path_neighbors += 1
            
    # A dead end has exactly one neighboring path
    return path_neighbors == 1

def count_dead_ends(grid, width, height):
    """Count the number of dead ends in the maze."""
    count = 0
    for y in range(height):
        for x in range(width):
            if is_dead_end(x, y, grid, width, height):
                count += 1
    return count


class CSPMazeGenerator:
    """A wrapper class for MazeCSP to fit into the game structure."""
    def __init__(self, width, height):
        # Ensure dimensions are odd to work with CSP
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.use_csp = True  # Flag to determine which method to use
        
    def generate_maze(self):
        """Generate a maze using CSP or backtracking based on the current mode."""
        if self.use_csp:
            # Try to solve with CSP
            maze_csp = MazeCSP(self.width, self.height)
            grid = maze_csp.solve()
            
            # If CSP succeeded, use that grid
            if grid is not None:
                start = (1, 1)
                end = (self.width - 2, self.height - 2)
                
                # Convert from list of lists to NumPy array if needed
                if not isinstance(grid, np.ndarray):
                    grid = np.array(grid)
                
                return grid, start, end
        
        # Either CSP failed or we're using backtracking
        grid = generate_maze_with_backtracking(self.width, self.height)
        start = (1, 1)
        end = (self.width - 2, self.height - 2)
        
        # Convert to NumPy array if not already
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
            
        return grid, start, end
    
    def toggle_method(self):
        """Toggle between CSP and backtracking methods."""
        self.use_csp = not self.use_csp
        return "CSP" if self.use_csp else "Backtracking"


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
        self.maze_generator = CSPMazeGenerator(MAZE_WIDTH, MAZE_HEIGHT)
        self.reset_game()
        self.running = True
        self.return_to_menu = False
        self.generation_method = "CSP"  # Default to CSP
    
    def reset_game(self):
        """Reset or initialize the game state"""
        self.maze, self.start, self.end = self.maze_generator.generate_maze()
        self.player_pos = self.start
        self.astar = AStar(self.maze)
        self.astar_path, self.visited_cells = self.astar.find_path(self.start, self.end)
        self.success = False
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
                elif event.key == pygame.K_c:
                    # Toggle between CSP and backtracking methods
                    self.generation_method = self.maze_generator.toggle_method()
                    print(f"Switched to {self.generation_method} generation method")
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
        
        # Draw generation method
        method_text = font.render(f"Method: {self.generation_method}", True, (255, 255, 255))
        screen.blit(method_text, (10, 5))
        
        # Draw steps counter
        steps_text = font.render(f"Steps: {self.steps_taken} (Optimal: {self.optimal_steps})", 
                                True, (255, 255, 255))
        screen.blit(steps_text, (10, 25))
        
        # Draw timer
        time_text = font.render(f"Time: {self.elapsed_time:.1f}s", True, (255, 255, 255))
        screen.blit(time_text, (SCREEN_WIDTH - 150, 5))
        
        # Draw legend
        legend_start = font.render("Green: Start", True, START_COLOR)
        screen.blit(legend_start, (SCREEN_WIDTH - 150, 25))
    
    def draw_controls_info(self):
        """Draw controls information in the bottom margin"""
        font = pygame.font.SysFont(None, 18)
        controls_text = font.render("Controls: WASD to move, R to reset, P toggle path, V toggle visited, C change method, ESC for menu", 
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