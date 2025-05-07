import numpy as np
import heapq
import matplotlib.pyplot as plt

# Initialize 10x10 grid (0: free, 1: obstacle)
np.random.seed(42)
grid = np.zeros((10, 10))
obstacles = np.random.choice([0, 1], size=(10, 10), p=[0.8, 0.2])
grid[1:9, 1:9] = obstacles[1:9, 1:9]  # Set obstacles
start = (0, 0)
goal = (9, 9)
grid[start] = 0
grid[goal] = 0

# A* algorithm
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_list = [(0, start)]  # (f_score, node)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}

    while open_list:
        current_f, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        neighbors = [(current[0] + dx, current[1] + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
        for neighbor in neighbors:
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + manhattan_distance(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
    return []

# Run A*
path = a_star(grid, start, goal)

# Visualize
plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap='binary')
if path:
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, 'r.-', label='Path')
plt.plot(start[1], start[0], 'go', label='Start')
plt.plot(goal[1], goal[0], 'b*', label='Goal')
plt.legend()
plt.title('Drone Path Optimization with A*')
plt.savefig('drone_path.png')
plt.show()

print("Path found:", path if path else "No path found")