import matplotlib.pyplot as plt
import pandas as pd
import heapq
import statistics
import time
from random import choice
import csv

shuffle_moves = 30

class PuzzleState:
    def __init__(self, board, goal, heuristic, moves=0, previous=None):
        self.board = board
        self.goal = goal
        self.heuristic = heuristic
        self.g = moves
        self.previous = previous
        self.before_previous = previous.previous if previous else None
        if self.heuristic == "h0":
            self.h = 0
        elif self.heuristic == "h1":
            self.h = self.misplaced_tiles()
        elif self.heuristic == "h2":
            self.h = self.manhattan_distance()
        else:
            self.h = 0  # default heuristic value if not specified
        self.priority = self.g + self.h

    def misplaced_tiles(self):
        count = 0
        flat_board = [num for row in self.board for num in row]
        flat_goal = [num for row in self.goal for num in row]
        for i in range(len(flat_board)):
            if flat_board[i] != 0 and flat_board[i] != flat_goal[i]:
                count += 1
        return count

    def manhattan_distance(self):
        distance = 0
        goal_positions = {self.goal[i][j]: (i, j) for i in range(3) for j in range(3)}
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:
                    goal_x, goal_y = goal_positions[self.board[i][j]]
                    distance += abs(i - goal_x) + abs(j - goal_y)
        return distance

    def is_goal(self):
        return self.board == self.goal

    def get_neighbors(self):
        neighbors = []
        x, y = next(
            (i, j)
            for i, row in enumerate(self.board)
            for j, val in enumerate(row)
            if val == 0
        )

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_board = [row[:] for row in self.board]
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]

                if self.before_previous and self.before_previous.board == new_board:
                    continue  # Ignore move that returns to the state two moves ago

                neighbor = PuzzleState(new_board, self.goal, self.heuristic, self.g + 1, self)
                neighbors.append(neighbor)

        return neighbors

    def __lt__(self, other):
        return self.priority < other.priority

def print_solution(solution):
    path = []
    state = solution
    while state:
        path.append(state.board)
        state = state.previous
    path.reverse()

def shuffle_puzzle(goal):
    current_board = [row[:] for row in goal]
    x, y = 1, 1  # The empty space starts in the center

    previous_board = None
    before_previous_board = None

    for _ in range(shuffle_moves):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid_moves = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_board = [row[:] for row in current_board]
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]

                if before_previous_board and new_board == before_previous_board:
                    continue  # Ignore move that returns to the state two moves ago

                valid_moves.append((nx, ny))

        if valid_moves:
            nx, ny = choice(valid_moves)
            before_previous_board = previous_board
            previous_board = [row[:] for row in current_board]
            current_board[x][y], current_board[nx][ny] = current_board[nx][ny], current_board[x][y]
            x, y = nx, ny

    return current_board

def solve_puzzle(start, goal, result, heuristic):
    start_time = time.time()
    count_NoM = 0  # Initialize the count of number of moves for this run

    start_state = PuzzleState(start, goal, heuristic)
    open_list = []
    heapq.heappush(open_list, (start_state.priority, start_state))
    closed_list = set()

    while open_list:
        _, current_state = heapq.heappop(open_list)

        if current_state.is_goal():
            end_time = time.time()
            # print_solution(current_state)
            # print(f"Solution found! Time taken: {end_time - start_time:.5f} seconds")
            result[0].append(float(end_time - start_time))
            result[1].append(float(current_state.g))
            result[2].append(float(count_NoM))
            return

        closed_list.add(tuple(map(tuple, current_state.board)))

        for neighbor in current_state.get_neighbors():
            count_NoM += 1  # Increment the count of number of moves
            neighbor_board_tuple = tuple(map(tuple, neighbor.board))
            if neighbor_board_tuple in closed_list:
                continue

            for i, (priority, existing) in enumerate(open_list):
                if neighbor.board == existing.board and neighbor.priority < priority:
                    open_list[i] = open_list[-1]
                    open_list.pop()
                    heapq.heapify(open_list)
                    break
            else:
                heapq.heappush(open_list, (neighbor.priority, neighbor))

    end_time = time.time()
    print(f"No solution found. Time taken: {end_time - start_time:.5f} seconds")

def iddfs(start, goal, result):
    def dls(node, depth, count_NoM):
        if node.is_goal():
            return node, count_NoM
        if depth == 0:
            return None, count_NoM
        for neighbor in node.get_neighbors():
            count_NoM += 1
            found, count_NoM = dls(neighbor, depth - 1, count_NoM)
            if found:
                return found, count_NoM
        return None, count_NoM

    start_time = time.time()
    depth = 0
    count_NoM = 0  # Initialize the count of number of moves for this run
    while True:
        start_state = PuzzleState(start, goal, "h0")
        found, count_NoM = dls(start_state, depth, count_NoM)
        if found:
            end_time = time.time()
            print_solution(found)
            # print(f"Solution found! Time taken: {end_time - start_time:.5f} seconds")
            result[0].append(float(end_time - start_time))
            result[1].append(float(depth))
            result[2].append(float(count_NoM))
            return
        depth += 1

# Collect results for all runs
h0_results = [[], [], []]
h1_results = [[], [], []]
h2_results = [[], [], []]
ids_results = [[], [], []]

# goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
goal = [[1, 2, 3], [4, 0, 5], [6, 7, 8]]

for i in range(50):
    # Shuffle the puzzle once
    start = shuffle_puzzle(goal)

    # Solve the 8-puzzle using A* with different heuristics
    solve_puzzle(start, goal, h0_results, heuristic="h0")
    solve_puzzle(start, goal, h1_results, heuristic="h1")
    solve_puzzle(start, goal, h2_results, heuristic="h2")

    # Solve the 8-puzzle using IDDFS
    iddfs(start, goal, ids_results)

    print("number", i, "done")

def print_results(heuristic, results):
    print(f"{heuristic} Results")
    print("Average Time:", statistics.mean(results[0]))
    print("Average Depth:", statistics.mean(results[1]))
    print("Depth Variance:", statistics.pvariance(results[1]))
    print("Average Moves:", statistics.mean(results[2]))
    print("Moves Variance:", statistics.pvariance(results[2]))

print_results("h0", h0_results)
print_results("h1", h1_results)
print_results("h2", h2_results)
print_results("IDDFS", ids_results)

print('h0', h0_results)
print('h1', h1_results)
print('h2', h2_results)
print('IDDFS', ids_results)


# Save results to CSV
def save_results_to_csv(filename, h0_results, h1_results, h2_results, ids_results):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(["Heuristic", "Time (s)", "Depth", "Number of Moves"])
        
        # Write h0 results
        for i in range(len(h0_results[0])):
            writer.writerow(["h0", h0_results[0][i], h0_results[1][i], h0_results[2][i]])
        
        # Write h1 results
        for i in range(len(h1_results[0])):
            writer.writerow(["h1", h1_results[0][i], h1_results[1][i], h1_results[2][i]])
        
        # Write h2 results
        for i in range(len(h2_results[0])):
            writer.writerow(["h2", h2_results[0][i], h2_results[1][i], h2_results[2][i]])
        
        # Write IDDFS results
        for i in range(len(ids_results[0])):
            writer.writerow(["IDDFS", ids_results[0][i], ids_results[1][i], ids_results[2][i]])

save_results_to_csv('puzzle_results_30.csv', h0_results, h1_results, h2_results, ids_results)
