import heapq
import time
import random
import csv

# --- パラメータ ---
shuffle_moves = 30  # シャッフル回数
num_puzzles = 100  # 解くパズルの数
time_limit = 720  # 15分 = 900秒

# --- 定数 ---
goal_state = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 0)
)
goal_positions = {
    tile: (i, j)
    for i, row in enumerate(goal_state)
    for j, tile in enumerate(row)
    if tile != 0
}

# --- 関数定義 ---
def print_board(board):
    """盤面を表示する."""
    for row in board:
        print(row)
    print("-" * 10)

def get_neighbors(state):
    """隣接状態を取得する。"""
    neighbors = []
    empty_i, empty_j = next(
        (i, j) for i, row in enumerate(state) for j, val in enumerate(row) if val == 0
    )
    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        ni, nj = empty_i + di, empty_j + dj
        if 0 <= ni < 4 and 0 <= nj < 4:
            new_state = list(list(row) for row in state)
            new_state[empty_i][empty_j], new_state[ni][nj] = new_state[ni][nj], new_state[empty_i][empty_j]
            neighbors.append(tuple(tuple(row) for row in new_state))
    return neighbors

def manhattan_distance(state):
    """マンハッタン距離を計算する。
    事前に計算したゴールポジションを使って高速化."""
    distance = 0
    for i in range(4):
        for j in range(4):
            tile = state[i][j]
            if tile != 0:
                goal_i, goal_j = goal_positions[tile]
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance

def solve_puzzle(start, goal, heuristic):
    """A*アルゴリズムを使用してパズルを解く.
    15分以上経過したらNoneを返す."""
    start_time = time.time()
    count_NoM = 0  # Number of Moves
    open_list = []
    closed_list = set()

    heapq.heappush(open_list, (0, start, 0))  # (f, state, g)

    while open_list:
        # 15分以上経過したらループを抜ける
        if time.time() - start_time > time_limit:
            print("Time limit exceeded.")
            return None, None, None

        f, current_state, g = heapq.heappop(open_list)

        # 途中経過を表示 (必要なければコメントアウト)
        # print(f"Depth: {g}, Priority: {f}")
        # print_board(current_state)

        if current_state == goal:
            end_time = time.time()
            return end_time - start_time, g, count_NoM

        closed_list.add(current_state)

        for neighbor in get_neighbors(current_state):
            count_NoM += 1
            if neighbor in closed_list:
                continue

            # ヒューリスティクス関数を適用
            if heuristic == "h0":
                h = 0
            elif heuristic == "h1":
                h = sum([1 for i in range(4) for j in range(4) if neighbor[i][j] != goal[i][j]])
            elif heuristic == "h2":
                h = manhattan_distance(neighbor)
            else:
                raise ValueError(f"Invalid heuristic: {heuristic}")

            heapq.heappush(open_list, (h + g + 1, neighbor, g + 1))

    return None, None, None  # 解が見つからなかった場合

def shuffle_puzzle(goal):
    """パズルをシャッフルする."""
    state = list(list(row) for row in goal)
    for _ in range(shuffle_moves):
        neighbors = get_neighbors(state)
        state = random.choice(neighbors)
    return tuple(tuple(row) for row in state)

# --- メイン処理 ---

if __name__ == "__main__":
    heuristics = ["h0", "h1", "h2"]
    results = {h: [[], [], []] for h in heuristics}  # (time, depth, moves)

    for j in range(10):
        for i in range(num_puzzles):
            print(f"Solving puzzle {i+1}...")
            start_state = shuffle_puzzle(goal_state)
            for heuristic in heuristics:
                time_taken, depth, moves = solve_puzzle(start_state, goal_state, heuristic)

                # 解けた場合のみ結果を保存
                if time_taken is not None:
                    results[heuristic][0].append(time_taken)
                    results[heuristic][1].append(depth)
                    results[heuristic][2].append(moves)
                else:
                    print(f"Puzzle {i+1} with {heuristic} could not be solved within the time limit.")


        # 結果を表示
        for heuristic in heuristics:
            print(f"\n--- {heuristic} Results ---")
            if results[heuristic][0]:  # 結果があれば表示
                print(f"Average Time: {sum(results[heuristic][0]) / len(results[heuristic][0]):.5f}s")
                print(f"Average Depth: {sum(results[heuristic][1]) / len(results[heuristic][1]):.2f}")
                print(f"Average Moves: {sum(results[heuristic][2]) / len(results[heuristic][2]):.2f}")
            else:
                print("No results within the time limit.")

        # jの値に応じてCSVファイル名を変更
        csv_filename = f"puzzle_results_{j}-3.csv"

        # 結果をCSVファイルに保存
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Heuristic", "Time (s)", "Depth", "Moves"])
            for heuristic in heuristics:
                for i in range(len(results[heuristic][0])):
                    writer.writerow([heuristic, results[heuristic][0][i], results[heuristic][1][i], results[heuristic][2][i]])
