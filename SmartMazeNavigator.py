import pygame
import numpy as np
import random
import heapq
from collections import deque
import time
from constraint import Problem, AllDifferentConstraint

pygame.init()

# Dimensions for the maze
# Each cell is a square of size CELL_SIZE
# Maze dimensions should be odd to ensure walls around the edges
CELL_SIZE = 20
MAZE_WIDTH = 77
MAZE_HEIGHT = 35

# Margins for the screen
TOP_MARGIN = 50
BOTTOM_MARGIN = 50

# Screen dimensions
# The screen size is based on the maze size and the cell size
SCREEN_WIDTH = MAZE_WIDTH * CELL_SIZE
SCREEN_HEIGHT = MAZE_HEIGHT * CELL_SIZE + TOP_MARGIN + BOTTOM_MARGIN

# Colors
BG_COLOR = (0, 0, 0) # Black background
WALL_COLOR = (50, 50, 50) # Dark gray for walls
PATH_COLOR = (200, 200, 255) # Light blue for paths
PLAYER_COLOR = (255, 0, 0) # Red for first player
PLAYER2_COLOR = (0, 0, 255)  # Blue for second player
START_COLOR = (0, 255, 0) # Green for start
END_COLOR = (255, 255, 0) # Yellow for end
ASTAR_PATH_COLOR = (255, 100, 100) # Light red for A* path
VISITED_COLOR = (100, 100, 255, 128) # Semi-transparent blue for visited cells
ADVERSARY_COLOR = (0, 255, 255)  # Cyan for AI
# UI Colors
TITLE_COLOR = (255, 255, 255) 
BUTTON_COLOR = (100, 100, 200)
BUTTON_HOVER_COLOR = (150, 150, 255)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Smart Maze Navigator")
clock = pygame.time.Clock()

# Button Class
# This class handles the button creation, drawing, and interaction
# It includes hover effects and click detection
class Button:
    def __init__(self, x, y, width, height, text, font_size=30):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.SysFont(None, font_size)
        self.is_hovered = False
    
    # Draw the button on the screen
    def draw(self, surface):
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BUTTON_TEXT_COLOR, self.rect, 2)
        
        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    
    # Check if the mouse is hovering over the button
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered
    
    # Check if the button is clicked
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False

# Main Menu Class
# This class handles the main menu of the game
# It includes buttons for different game modes and a title
# The menu is displayed until the user selects a game mode
# The game mode is returned to the main game loop
class MainMenu:
    def __init__(self):
        self.running = True
        self.game_mode = None  # 'single', 'multi', or 'ai'
        
        # Create buttons
        button_width, button_height = 200, 60
        button_x = SCREEN_WIDTH // 2 - button_width // 2
        
        self.single_button = Button(
            button_x, SCREEN_HEIGHT // 2, 
            button_width, button_height, "Single Player"
        )
        
        self.multi_button = Button(
            button_x, SCREEN_HEIGHT // 2 + button_height + 20, 
            button_width, button_height, "Multiplayer"
        )
        
        self.ai_button = Button(
            button_x, SCREEN_HEIGHT // 2 + 2*(button_height + 20),
            button_width, button_height, "VS AI"
        )
        
        self.title_font = pygame.font.SysFont("arial", 72)
        self.background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.background.fill((20, 20, 50))
        
        for _ in range(30):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT)
            radius = random.randint(1, 3)
            pygame.draw.circle(self.background, (255, 255, 255, 50), (x, y), radius)
    
    # Handle events: mouse movement, button clicks, and quitting
    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        self.single_button.check_hover(mouse_pos)
        self.multi_button.check_hover(mouse_pos)
        self.ai_button.check_hover(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if self.single_button.is_clicked(mouse_pos, event):
                self.game_mode = 'single'
                self.running = False
            
            if self.multi_button.is_clicked(mouse_pos, event):
                self.game_mode = 'multi'
                self.running = False
                
            if self.ai_button.is_clicked(mouse_pos, event):
                self.game_mode = 'ai'
                self.running = False
    
    # Draw the menu: background, title, buttons
    def draw(self):
        screen.blit(self.background, (0, 0))
        
        #shadow
        title_text = "Smart Maze Navigator"
        shadow_surf = self.title_font.render(title_text, True, (0, 0, 0))
        shadow_rect = shadow_surf.get_rect(center=(SCREEN_WIDTH // 2 + 2, SCREEN_HEIGHT // 3 + 2))
        screen.blit(shadow_surf, shadow_rect)
        
        # Title
        title_surf = self.title_font.render("Smart Maze Navigator", True, TITLE_COLOR)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        screen.blit(title_surf, title_rect)
        
        # Subtitle
        subtitle_font = pygame.font.SysFont("arial", 24)
        subtitle_surf = subtitle_font.render("Choose your game mode", True, (200, 200, 255))
        subtitle_rect = subtitle_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3 + 40))
        screen.blit(subtitle_surf, subtitle_rect)
        
        # Buttons
        self.single_button.draw(screen)
        self.multi_button.draw(screen)
        self.ai_button.draw(screen)
        
        pygame.display.flip()
    
    # Main loop for the menu
    def run(self):
        while self.running:
            self.handle_events()
            self.draw()
            clock.tick(60)
        return self.game_mode

# Maze Generation using CSP
# This class uses the constraint satisfaction problem (CSP) approach to generate a maze
# It sets up variables for each cell, adds constraints for walls, paths, and dead ends
# The maze is generated by solving the CSP
class MazeCSP:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = None
        
    # Set up the CSP variables and constraints
    def setup_variables(self):
        self.problem = Problem()
        for y in range(self.height):
            for x in range(self.width):
                self.problem.addVariable((x, y), [0, 1])
                
    # Add constraints to the CSP
    def add_border_constraint(self):
        for y in range(self.height):
            for x in range(self.width):
                if x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1:
                    self.problem.addConstraint(lambda val: val == 1, [(x, y)])

    # Add entrance and exit constraints    
    def add_entrance_exit_constraint(self, entrance_x=1, entrance_y=0, exit_x=None, exit_y=None):
        if exit_x is None:
            exit_x = self.width - 2
        if exit_y is None:
            exit_y = self.height - 1
        self.problem.addConstraint(lambda val: val == 0, [(entrance_x, entrance_y)])
        self.problem.addConstraint(lambda val: val == 0, [(entrance_x, entrance_y + 1)])
        self.problem.addConstraint(lambda val: val == 0, [(exit_x, exit_y)])
        self.problem.addConstraint(lambda val: val == 0, [(exit_x, exit_y - 1)])
    
    # Add constraint to prevent 2x2 blocks of walls
    def add_no_2x2_wall_constraint(self):
        for y in range(self.height - 1):
            for x in range(self.width - 1):
                top_left = (x, y)
                top_right = (x + 1, y)
                bottom_left = (x, y + 1)
                bottom_right = (x + 1, y + 1)
                self.problem.addConstraint(
                    lambda tl, tr, bl, br: tl + tr + bl + br < 4,
                    [top_left, top_right, bottom_left, bottom_right]
                )
    
    # Checks if a cell is a dead end
    def is_dead_end(self, x, y, grid):
        if grid[y][x] != 0:
            return False
        neighbors = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1)
        ]
        path_neighbors = 0
        for nx, ny in neighbors:
            if 0 <= nx < self.width and 0 <= ny < self.height and grid[ny][nx] == 0:
                path_neighbors += 1
        return path_neighbors == 1
    
    # Count the number of dead ends in the maze
    def count_dead_ends(self, grid):
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.is_dead_end(x, y, grid):
                    count += 1
        return count
    
    # Check if the maze is connected
    def is_connected(self, grid, start=None, end=None):
        if start is None:
            for y in range(self.height):
                for x in range(self.width):
                    if grid[y][x] == 0:
                        start = (x, y)
                        break
                if start:
                    break
        if end is None:
            end = (self.width - 2, self.height - 2)
        visited = set()
        queue = [start]
        visited.add(start)
        while queue:
            x, y = queue.pop(0)
            if (x, y) == end:
                return True
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    grid[ny][nx] == 0 and (nx, ny) not in visited):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        all_paths = sum(row.count(0) for row in grid)
        return len(visited) == all_paths
    
    # Solve the CSP to generate a valid maze
    def solve(self, max_attempts=50):
        self.setup_variables()
        self.add_border_constraint()
        self.add_entrance_exit_constraint()
        self.add_no_2x2_wall_constraint()
        for _ in range(max_attempts):
            solution = self.problem.getSolution()
            if not solution:
                return None
            grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
            for (x, y), value in solution.items():
                grid[y][x] = value
            if not self.is_connected(grid):
                continue
            if hasattr(self, 'min_dead_ends'):
                dead_ends = self.count_dead_ends(grid)
                if dead_ends < self.min_dead_ends:
                    continue
            self.grid = grid
            return grid
        print("CSP failed to find a valid maze after max attempts.")
        return None
    
    # Get the start and end positions of the maze
    def get_start_end(self):
        return (1, 1), (self.width - 2, self.height - 2)

# Alternative maze generation using backtracking in case CSP fails
def generate_maze_with_backtracking(width, height, min_dead_ends=3):
    grid = np.ones((height, width), dtype=int)
    start_x = 1
    start_y = 1
    grid[start_y][start_x] = 0
    stack = [(start_x, start_y)]
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    while stack:
        x, y = stack[-1]
        unvisited = []
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 < nx < width-1 and 0 < ny < height-1 and grid[ny][nx] == 1):
                if grid[ny][nx] == 1:
                    unvisited.append((nx, ny, dx//2, dy//2))
        if unvisited:
            nx, ny, wall_x, wall_y = unvisited[0]
            grid[y + wall_y][x + wall_x] = 0
            grid[ny][nx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()
    grid[0][1] = 0
    grid[height-1][width-2] = 0
    dead_ends = count_dead_ends(grid, width, height)
    if dead_ends < min_dead_ends:
        attempts = 0
        while dead_ends < min_dead_ends and attempts < 100:
            x, y = random.randint(1, width-2), random.randint(1, height-2)
            if grid[y][x] == 0 and not is_dead_end(x, y, grid, width, height):
                path_neighbors = []
                for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                    if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 0:
                        path_neighbors.append((nx, ny))
                if len(path_neighbors) >= 3:
                    for i in range(len(path_neighbors) - 1):
                        nx, ny = path_neighbors[i]
                        grid[ny][nx] = 1
                    dead_ends = count_dead_ends(grid, width, height)
            attempts += 1
    return grid

# Check if a cell is a dead end
def is_dead_end(x, y, grid, width, height):
    if grid[y][x] != 0:
        return False
    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    path_neighbors = 0
    for nx, ny in neighbors:
        if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 0:
            path_neighbors += 1
    return path_neighbors == 1

# Count the number of dead ends in the maze
def count_dead_ends(grid, width, height):
    count = 0
    for y in range(height):
        for x in range(width):
            if is_dead_end(x, y, grid, width, height):
                count += 1
    return count

# Main maze generation class
# This class handles the maze generation process
# It uses either CSP or backtracking methods based on the user's choice
# The maze is generated and returned along with start and end positions
class CSPMazeGenerator:
    def __init__(self, width, height):
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.use_csp = True
        
    # Generate a maze using the selected method
    def generate_maze(self):
        if self.use_csp:
            maze_csp = MazeCSP(self.width, self.height)
            grid = maze_csp.solve()
            if grid is not None:
                start = (1, 1)
                end = (self.width - 2, self.height - 2)
                if not isinstance(grid, np.ndarray):
                    grid = np.array(grid)
                return grid, start, end
        grid = generate_maze_with_backtracking(self.width, self.height)
        start = (1, 1)
        end = (self.width - 2, self.height - 2)
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
        return grid, start, end
    
    # Toggle between CSP and backtracking methods
    def toggle_method(self):
        self.use_csp = not self.use_csp
        return "CSP" if self.use_csp else "Backtracking"

##### Bushra's Class 
class AStar:
    def __init__(self, maze):
        self.maze = maze   #making maze instance
        self.height, self.width = maze.shape   #getting dimensions 
    
    def update_maze(self, maze):
        """Update the maze after it has been changed"""
        self.maze = maze      #update maze instance
        self.height, self.width = maze.shape   #update dimensions 
    
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])    #manhattan distance |x1-x2| + |y1 - y2|
    
    def get_neighbors(self, node):    #getting valid neighbouring nodes for example left,right,up,down that are possible
        x, y = node    #taking the current nodes coordinates
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:   #for checking four possible directions
            nx, ny = x + dx, y + dy   #calcuting the coordinates of the neighbour
            if 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny][nx] == 0:   #checking if the neighbor node valid and we can move there
                neighbors.append((nx, ny))  #if valid then add to list
        return neighbors
    
    def find_path(self, start, end):   #to find shortest path through A*
        open_set = []            #priority queue
        heapq.heappush(open_set, (0, start))    # score, node
        came_from = {}                #tracks best path
        g_score = {start: 0}            #cost from start to node
        f_score = {start: self.heuristic(start, end)}   #cost(g_score)  + heuristic
        open_set_hash = {start}    #quick lookup of nodes in openset
        visited = set()    #track of all visited nodes
        
        while open_set:
            _, current = heapq.heappop(open_set)   #getting the node that has the lowest f_score
            open_set_hash.remove(current)
            visited.add(current)
            
            if current == end:   #so that if we have reached the goal, we return the path
                path = []
                while current in came_from:   #backtracking
                    path.append(current)
                    current = came_from[current]
                path.append(start)   #adding start node
                path.reverse()    #reversing to get correct order
                return path, visited
            
            for neighbor in self.get_neighbors(current):   #to chekc all neighbours of the current node
                tentative_g_score = g_score[current] + 1   #to calculate the g_score of the neighbor node
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:  #so that if this path is better than the previous one
                    came_from[neighbor] = current    #update the best path
                    g_score[neighbor] = tentative_g_score   #update the g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, end)  #update the f_score
                    
                    if neighbor not in open_set_hash:   #if neighbor not in open_set update it 
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return [], visited   #if goal is not reached or open set is empty

### making dis class for ai mode
### it runs via min max as well as astarr adn uses the best possible for each move
### it makes sure ai wins and when player near end it favors ai in all possible ways to make sure ai winss
class AIPlayer:
    def __init__(self, maze, position, opponent_position):
        self.maze = maze
        self.position = position
        self.opponent_position = opponent_position
        self.astar = AStar(maze)
        self.depth_limit = 3  #dis one shows how many moves it looks forwardd
        self.end = None  #dis depends and shows where to end when ai donee
        self.path_to_goal = []  # checkss which calculated to patthh
        self.path_recalculate_counter = 0  # checksss best path each timee
        
    def calculate_path_to_goal(self):   ## dis function checks and fijnds whatevva path to take to endd
        """Calculate direct path to goal using A*"""
        if self.end:
            path, _ = self.astar.find_path(self.position, self.end)
            print("A* Being Used")
            self.path_to_goal = path
            return path
        return []
        
    def get_best_move(self):
        """Get the next best move for the AI"""
        # everytimee dis chwcks which is the best move from noww
        self.path_recalculate_counter += 1
        if not self.path_to_goal or self.path_recalculate_counter > 5:
            self.calculate_path_to_goal()
            self.path_recalculate_counter = 0
        
        
        if len(self.path_to_goal) >= 2:
            # findign second best one option that is to be takenn
            for i, pos in enumerate(self.path_to_goal):
                if pos == self.position and i + 1 < len(self.path_to_goal):
                    return self.path_to_goal[i + 1]
        
        return self.get_minimax_move()

    ##using minmaxx fro movess
    def get_minimax_move(self):
        """Use minimax to find the best move (original method)"""
        print("Alpha Beta Being Used")
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in self.get_possible_moves(self.position):
            value = self.minimax(
                self.depth_limit, 
                False, 
                alpha, 
                beta, 
                move, 
                self.opponent_position
            )
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
        
        return best_move


     ##using minmaxx fro moving forward or backward whichever bestt
    def minimax(self, depth, is_maximizing, alpha, beta, current_pos, opponent_pos):
        if depth == 0 or self.is_terminal_state(current_pos, opponent_pos):
            return self.evaluate_state(current_pos, opponent_pos)
            
        if is_maximizing:
            max_eval = float('-inf')
            for move in self.get_possible_moves(current_pos):
                eval = self.minimax(depth-1, False, alpha, beta, move, opponent_pos)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_possible_moves(opponent_pos):
                eval = self.minimax(depth-1, True, alpha, beta, current_pos, move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def get_possible_moves(self, pos):
        x, y = pos
        moves = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(self.maze[0]) and 0 <= ny < len(self.maze):
                if self.maze[ny][nx] == 0:  # onli valid moves
                    moves.append((nx, ny))
        return moves
    
    def is_terminal_state(self, ai_pos, player_pos):
       ##for stoppin when reahced endd
        return ai_pos == self.end or player_pos == self.end
        
    def evaluate_state(self, ai_pos, player_pos):
        ##favoring aii when player clojee
        ai_dist = self.astar.heuristic(ai_pos, self.end)
        player_dist = self.astar.heuristic(player_pos, self.end)
        return player_dist - ai_dist  ##higher is better for AI

### Bushra's Class with a mix of all of ours codes. 
class Game:
    def __init__(self, game_mode='single'):
        self.maze_gen = CSPMazeGenerator(MAZE_WIDTH, MAZE_HEIGHT)
        self.game_mode = game_mode
        self.generation_method = "CSP"
        self.adversary_pos = None
        self.ai_player = None
        self.player_turn = True
        self.ai_move_timer = 0
        self.ai_move_delay = 100 # AI move delay in milliseconds
        self.ai_reached_end = False
        self.reset_game()
        self.running = True
        self.return_to_menu = False
    
    # Reset the game state
    # This function generates a new maze and resets player positions
    # It also initializes the AI player if the game mode is 'ai'
    # The function sets the start and end positions for the maze
    def reset_game(self):
        self.maze, self.start, self.end = self.maze_gen.generate_maze()
        self.player_pos = self.start
        if self.game_mode == 'ai':
            self.adversary_pos = (self.start[0] + 2, self.start[1])
            if self.maze[self.adversary_pos[1]][self.adversary_pos[0]] == 1:
                for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
                    nx, ny = self.start[0] + dx, self.start[1] + dy
                    if self.maze[ny][nx] == 0:
                        self.adversary_pos = (nx, ny)
                        break
            self.ai_player = AIPlayer(self.maze, self.adversary_pos, self.player_pos)
            self.ai_player.end = self.end
        self.ai_move_timer = pygame.time.get_ticks()
        self.ai_reached_end = False
        self.astar = AStar(self.maze)
        self.astar_path, self.visited_cells = self.astar.find_path(self.start, self.end)
        self.show_astar_path = True
        self.show_visited = True
        if self.game_mode == 'multi':
            self.player2_pos = self.start
            self.player2_steps = 0
            self.player2_success = False
        self.success = False
        self.steps_taken = 0
        self.optimal_steps = len(self.astar_path) - 1
        self.timer_start = time.time()
        self.elapsed_time = 0
    
    # Handle events: key presses, mouse clicks, etc.
    # This function checks for events and updates the game state accordingly
    # It includes controls for player movement, game reset, and quitting
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                # Common controls
                if event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_p:
                    # Toggle A* path visibility
                    self.show_astar_path = not self.show_astar_path
                elif event.key == pygame.K_v:
                    # Toggle visited cells visibility
                    self.show_visited = not self.show_visited
                elif event.key == pygame.K_n:
                    self.reset_game()
                elif event.key == pygame.K_ESCAPE:
                    self.return_to_menu = True
                    self.running = False
                elif event.key == pygame.K_c:
                    # Toggle between CSP and backtracking methods
                    self.generation_method = self.maze_gen.toggle_method()
                    print(f"Switched to {self.generation_method} generation method")
                    self.reset_game()
                if not self.success and (self.game_mode != 'multi' or not self.player2_success):
                    self.handle_player_movement(event)

    # Handle player movement
    # This function checks the key pressed and updates the player position
    # It also checks if the player has reached the end of the maze
    def handle_player_movement(self, event):
        if self.game_mode != 'multi' or not self.success:
            x, y = self.player_pos
            new_x, new_y = x, y
            if event.key == pygame.K_w: new_x, new_y = x, y-1
            elif event.key == pygame.K_s: new_x, new_y = x, y+1
            elif event.key == pygame.K_a: new_x, new_y = x-1, y
            elif event.key == pygame.K_d: new_x, new_y = x+1, y
            if self.is_valid_move(new_x, new_y):
                self.player_pos = (new_x, new_y)
                self.steps_taken += 1
                if self.player_pos == self.end:
                    self.success = True
                    self.elapsed_time = time.time() - self.timer_start
                if self.game_mode == 'ai' and hasattr(self, 'ai_player'):
                    self.ai_player.opponent_position = self.player_pos
        if self.game_mode == 'multi' and not self.player2_success:
            x, y = self.player2_pos
            new_x, new_y = x, y
            if event.key == pygame.K_UP: new_x, new_y = x, y-1
            elif event.key == pygame.K_DOWN: new_x, new_y = x, y+1
            elif event.key == pygame.K_LEFT: new_x, new_y = x-1, y
            elif event.key == pygame.K_RIGHT: new_x, new_y = x+1, y
            if self.is_valid_move(new_x, new_y):
                self.player2_pos = (new_x, new_y)
                self.player2_steps += 1
                if self.player2_pos == self.end:
                    self.player2_success = True
    
    # Check if the move is valid
    # This function checks if the new position is within the maze bounds
    # and if the cell is not a wall (1)
    def is_valid_move(self, x, y):
        return (0 <= x < MAZE_WIDTH and 
                0 <= y < MAZE_HEIGHT and 
                self.maze[y][x] == 0)

    # Update the game state
    # This function is called to update the game state
    # It includes checking for player movements, AI moves, and game success conditions
    # The function also handles the timing for AI moves
    # The AI moves every ai_move_delay milliseconds
    def update(self):
        if not self.success:
            self.elapsed_time = time.time() - self.timer_start
        if self.game_mode == 'ai' and not self.success:
            current_time = pygame.time.get_ticks()
            if current_time - self.ai_move_timer >= self.ai_move_delay:
                best_move = self.ai_player.get_best_move()
                if best_move:
                    self.adversary_pos = best_move
                    self.ai_player.position = best_move
                    if self.adversary_pos == self.end and not self.ai_reached_end:
                        self.ai_reached_end = True
                self.ai_move_timer = current_time

    # Draw the maze and game elements
    # This function is called to draw the maze, player, and other elements
    # It includes the maze grid, player positions, and any additional UI elements
    # The function uses the Pygame library to render the graphics
    # The maze is drawn using rectangles for walls and paths
    def draw(self):
        screen.fill(BG_COLOR)
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
                if self.maze[y][x] == 1:
                    pygame.draw.rect(screen, WALL_COLOR, rect)
                else:
                    pygame.draw.rect(screen, PATH_COLOR, rect)
        if self.show_astar_path and self.show_visited:
            for x, y in self.visited_cells:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, VISITED_COLOR, rect)
        if self.show_astar_path:
            for x, y in self.astar_path:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, ASTAR_PATH_COLOR, rect)
        start_rect = pygame.Rect(self.start[0]*CELL_SIZE, self.start[1]*CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
        end_rect = pygame.Rect(self.end[0]*CELL_SIZE, self.end[1]*CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, START_COLOR, start_rect)
        pygame.draw.rect(screen, END_COLOR, end_rect)
        player_rect = pygame.Rect(self.player_pos[0]*CELL_SIZE, self.player_pos[1]*CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, PLAYER_COLOR, player_rect)
        if self.game_mode == 'multi':
            player2_rect = pygame.Rect(self.player2_pos[0]*CELL_SIZE, self.player2_pos[1]*CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, PLAYER2_COLOR, player2_rect)
        elif self.game_mode == 'ai':
            ai_rect = pygame.Rect(self.adversary_pos[0]*CELL_SIZE, self.adversary_pos[1]*CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, ADVERSARY_COLOR, ai_rect)
        self.draw_ui()
        if not self.success:
            self.elapsed_time = time.time() - self.timer_start
        self.draw_controls_info()
        if self.success:
            self.draw_success_popup()

        pygame.display.flip()
    
    # Draw controls information in the bottom margin
    # This function is called to display the controls for the game
    # It includes the controls for player movement and other actions
    def draw_controls_info(self):
        font = pygame.font.SysFont(None, 18)
        controls_text = font.render(" ", 
                                   True, (255, 255, 255))
        screen.blit(controls_text, (10, SCREEN_HEIGHT - 30))
    
    # Draw success popup
    # This function is called when the player reaches the end of the maze
    # It displays the success message and relevant stats
    def draw_success_popup(self):
        s_width, s_height = 400, 200
        surf = pygame.Surface((s_width, s_height), pygame.SRCALPHA)
        surf.fill((0, 0, 100, 200))
        screen.blit(surf, (SCREEN_WIDTH//2 - s_width//2, SCREEN_HEIGHT//2 - s_height//2))
        font = pygame.font.SysFont(None, 36)
        if self.game_mode == 'multi':
            if self.success and self.player2_success:
                title = "Tie Game!"
            elif self.success:
                title = "Player 1 Wins!"
            else:
                title = "Player 2 Wins!"
        elif self.game_mode == 'ai':
            if self.ai_reached_end and self.success:
                title = "You Made It!"
            elif self.ai_reached_end:
                title = "Keep Going!"
            else:
                title = "You Win!"
        else:
            title = "Success!"
        text = font.render(title, True, (255,255,255))
        screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - 70))
        font = pygame.font.SysFont(None, 24)
        stats = [
            f"Time: {self.elapsed_time:.1f}s",
            f"Player 1 steps: {self.steps_taken}"
        ]
        if self.game_mode == 'multi':
            stats.append(f"Player 2 steps: {self.player2_steps}")
        elif self.game_mode == 'ai' and self.ai_reached_end:
            stats.append(f"AI reached the end!")
        if self.game_mode != 'multi':
            efficiency = self.optimal_steps / max(1, self.steps_taken) * 100
            stats.append(f"Efficiency: {efficiency:.1f}%")
        for i, stat in enumerate(stats):
            text = font.render(stat, True, (255,255,255))
            screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - 30 + i*25))
        optimal_text = font.render(f"Optimal path: {self.optimal_steps} steps", True, (255, 255, 255))
        optimal_rect = optimal_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        screen.blit(optimal_text, optimal_rect)
        text = font.render("Press R to restart or ESC for menu", True, (255,255,255))
        screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 + 75))

    # Draw the UI elements
    # This function is called to display the game status
    # It includes the current mode, steps taken, time elapsed, and controls
    # It also displays the success message if the player has reached the end
    def draw_ui(self):
        font = pygame.font.SysFont(None, 22)
        mode_text = f"Mode: {self.game_mode.upper()}"
        method_text = font.render(f"                              Method: {self.generation_method}", True, (255, 255, 255))
        screen.blit(method_text, (10, 10))
        if self.game_mode == 'multi':
            steps_text = f"P1: {self.steps_taken} | P2: {self.player2_steps}"
        else:
            steps_text = f"Steps: {self.steps_taken} (Optimal: {self.optimal_steps})"
        time_text = f"Time: {self.elapsed_time:.1f}s"
        screen.blit(font.render(mode_text, True, (255,255,255)), (10, 10))
        screen.blit(font.render(steps_text, True, (255,255,255)), (10, 35))
        screen.blit(font.render(time_text, True, (255,255,255)), (SCREEN_WIDTH-150, 10))
        controls_font = pygame.font.SysFont(None, 20)
        controls = [
            "WASD: Move Player 1",
            "Arrows: Move Player 2" if self.game_mode == 'multi' else "",
            "R: Reset | P: Toggle Path | V: Toggle Visited | C: change method",
            "N: New Maze | ESC: Menu"
        ]
        for i, control in enumerate(controls):
            if control:
                screen.blit(controls_font.render(control, True, (255,255,255)), 
                           (10, SCREEN_HEIGHT - 30 - i*20))
        if self.success or (self.game_mode == 'multi' and self.player2_success):
            self.draw_success()
    
    # Draw success message
    # This function is called when the player reaches the end of the maze
    # It displays the success message and relevant stats
    # It also provides options to restart the game or return to the menu
    def draw_success(self):
        s_width, s_height = 400, 200
        surf = pygame.Surface((s_width, s_height), pygame.SRCALPHA)
        surf.fill((0, 0, 100, 200))
        screen.blit(surf, (SCREEN_WIDTH//2 - s_width//2, SCREEN_HEIGHT//2 - s_height//2))
        font = pygame.font.SysFont(None, 36)
        if self.game_mode == 'multi':
            if self.success and self.player2_success:
                title = "Tie Game!"
            elif self.success:
                title = "Player 1 Wins!"
            else:
                title = "Player 2 Wins!"
        else:
            title = "Success!"
        text = font.render(title, True, (255,255,255))
        screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - 70))
        font = pygame.font.SysFont(None, 24)
        stats = [
            f"Time: {self.elapsed_time:.1f}s",
            f"Player 1 steps: {self.steps_taken}"
        ]
        if self.game_mode == 'multi':
            stats.append(f"Player 2 steps: {self.player2_steps}")
        else:
            efficiency = self.optimal_steps / max(1, self.steps_taken) * 100
            stats.append(f"Efficiency: {efficiency:.1f}%")
        for i, stat in enumerate(stats):
            text = font.render(stat, True, (255,255,255))
            screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - 30 + i*25))
        optimal_text = font.render(f"Optimal path: {self.optimal_steps} steps", True, (255, 255, 255))
        optimal_rect = optimal_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        screen.blit(optimal_text, optimal_rect)
        text = font.render("Press R to restart or ESC for menu", True, (255,255,255))
        screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 + 50))
    
    # Main game loop
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            clock.tick(60)
        
        return self.return_to_menu

# Main function to run the game
def main():
    """Entry point for the game"""
    pygame.display.set_caption("Smart Maze Navigator")
    
    running = True
    while running:
        menu = MainMenu()
        game_mode = menu.run()
        
        if game_mode:
            game = Game(game_mode)
            return_to_menu = game.run()
            
            if not return_to_menu:
                running = False
        else:
            running = False
    
    pygame.quit()

# Initialize Pygame
if __name__ == "__main__":
    main()
