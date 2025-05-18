from collections import deque
import heapq
import math
import random
import time

moves = {"L": (0, -1), "R": (0, 1), "U": (-1, 0), "D": (1, 0)}

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def is_valid(x, y):
    return 0 <= x < 3 and 0 <= y < 3

def state_to_tuple(state):
    return tuple(map(tuple, state))

def move_tile(state, move):
    x, y = find_blank(state)
    try:
        dx, dy = moves[move]
    except KeyError:
        return None
    nx, ny = x + dx, y + dy
    if is_valid(nx, ny):
        new_state = [row[:] for row in state]
        new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
        return new_state
    return None

def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                for gi in range(3):
                    for gj in range(3):
                        if goal_state[gi][gj] == value:
                            distance += abs(i - gi) + abs(j - gj)
                            break
    return distance

def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = {state_to_tuple(start_state)}
    while queue:
        state, path = queue.popleft()
        if state == goal_state:
            return path, 1.0
        for move in moves:
            new_state = move_tile(state, move)
            if new_state:
                tup = state_to_tuple(new_state)
                if tup not in visited:
                    visited.add(tup)
                    queue.append((new_state, path + [move]))
    return None, 0.0

def ucs(start_state, goal_state):
    pq = [(0, start_state, [])]
    visited = set()
    while pq:
        cost, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path, 1.0
        state_tuple = state_to_tuple(state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for move in moves:
            new_state = move_tile(state, move)
            if new_state:
                tup = state_to_tuple(new_state)
                if tup not in visited:
                    heapq.heappush(pq, (cost + 1, new_state, path + [move]))
    return None, 0.0

def dfs(start_state, goal_state, max_depth=100):
    start_time = time.time()
    timeout = 10
    stack = [(start_state, [], 0)]
    visited = set()
    while stack and time.time() - start_time < timeout:
        state, path, depth = stack.pop()
        if state == goal_state:
            return path
        if depth >= max_depth:
            continue
        state_tuple = state_to_tuple(state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for move in moves:
            new_state = move_tile(state, move)
            if new_state:
                tup = state_to_tuple(new_state)
                if tup not in visited:
                    stack.append((new_state, path + [move], depth + 1))
    return None

def ids(start_state, goal_state):
    def dls(state, depth, path, visited):
        if state == goal_state:
            return path
        if depth == 0:
            return None
        visited.add(state_to_tuple(state))
        for move in moves:
            new_state = move_tile(state, move)
            if new_state:
                tup = state_to_tuple(new_state)
                if tup not in visited:
                    result = dls(new_state, depth - 1, path + [move], visited)
                    if result is not None:
                        return result
        return None

    depth = 0
    max_depth = 100
    while depth <= max_depth:
        visited = set()
        result = dls(start_state, depth, [], visited)
        if result is not None:
            return result, 1.0
        depth += 1
    return None, 0.0

def gbfs(start_state, goal_state):
    pq = [(manhattan_distance(start_state, goal_state), start_state, [])]
    visited = set()
    while pq:
        _, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path, 1.0
        state_tuple = state_to_tuple(state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for move in moves:
            new_state = move_tile(state, move)
            if new_state:
                tup = state_to_tuple(new_state)
                if tup not in visited:
                    heapq.heappush(pq, (manhattan_distance(new_state, goal_state), new_state, path + [move]))
    return None, 0.0

def a_star_search(start_state, goal_state):
    pq = [(manhattan_distance(start_state, goal_state), 0, start_state, [])]
    visited = set()
    while pq:
        _, cost, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path, 1.0
        state_tuple = state_to_tuple(state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for move in moves:
            new_state = move_tile(state, move)
            if new_state:
                tup = state_to_tuple(new_state)
                if tup not in visited:
                    new_cost = cost + 1
                    priority = new_cost + manhattan_distance(new_state, goal_state)
                    heapq.heappush(pq, (priority, new_cost, new_state, path + [move]))
    return None, 0.0

def ida_star(start_state, goal_state):
    bound = manhattan_distance(start_state, goal_state)
    opp = {"L": "R", "R": "L", "U": "D", "D": "U"}
    def search(state, g, bound, path):
        f = g + manhattan_distance(state, goal_state)
        if f > bound:
            return f, None
        if state == goal_state:
            return f, path
        min_cost = math.inf
        for move in moves:
            if path and move == opp.get(path[-1]):
                continue
            new_state = move_tile(state, move)
            if new_state:
                result, res_path = search(new_state, g + 1, bound, path + [move])
                if res_path is not None:
                    return result, res_path
                if result < min_cost:
                    min_cost = result
        return min_cost, None

    while True:
        result, path = search(start_state, 0, bound, [])
        if path is not None:
            return path, 1.0
        if result == math.inf:
            return None, 0.0
        bound = result

def simple_hill_climbing(start_state, goal_state):
    current_state = [row[:] for row in start_state]
    path = []
    while True:
        best_distance = manhattan_distance(current_state, goal_state)
        improved = False
        for move in moves:
            new_state = move_tile(current_state, move)
            if new_state:
                new_distance = manhattan_distance(new_state, goal_state)
                if new_distance < best_distance:
                    current_state = new_state
                    path.append(move)
                    improved = True
                    break
        if not improved:
            confidence = 1 / (1 + best_distance)
            return path, confidence

def steepest_ascent_hill_climbing(start_state, goal_state):
    current_state = [row[:] for row in start_state]
    path = []
    while True:
        best_distance = manhattan_distance(current_state, goal_state)
        best_move = None
        best_state = None
        for move in moves:
            new_state = move_tile(current_state, move)
            if new_state:
                new_distance = manhattan_distance(new_state, goal_state)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_state = new_state
                    best_move = move
        if best_state is None:
            confidence = 1 / (1 + manhattan_distance(current_state, goal_state))
            return path, confidence
        current_state = best_state
        path.append(best_move)

def stochastic_hill_climbing(start_state, goal_state):
    current_state = [row[:] for row in start_state]
    path = []
    while True:
        neighbors = []
        current_distance = manhattan_distance(current_state, goal_state)
        for move in moves:
            new_state = move_tile(current_state, move)
            if new_state:
                new_distance = manhattan_distance(new_state, goal_state)
                if new_distance < current_distance:
                    neighbors.append((new_state, move))
        if not neighbors:
            confidence = 1 / (1 + manhattan_distance(current_state, goal_state))
            return path, confidence
        new_state, move = random.choice(neighbors)
        current_state = new_state
        path.append(move)

def simulated_annealing(start_state, goal_state):
    current_state = [row[:] for row in start_state]
    best_state = [row[:] for row in start_state]
    path = []
    best_path = []
    temperature = 1000.0
    min_temperature = 0.01
    max_iterations = 10000
    cooling_rate = 0.995
    max_cost = manhattan_distance(start_state, goal_state) + 10
    current_cost = manhattan_distance(current_state, goal_state)
    best_cost = current_cost
    step_count = 0
    start_time = time.time()

    for iteration in range(max_iterations):
        if time.time() - start_time > 10:
            break
        step_count += 1
        if current_state == goal_state:
            return path, 1.0, step_count
        possible_moves = []
        x, y = find_blank(current_state)
        for move, (dx, dy) in moves.items():
            if is_valid(x + dx, y + dy):
                possible_moves.append(move)
        if not possible_moves:
            break
        move = random.choice(possible_moves)
        new_state = move_tile(current_state, move)
        new_cost = manhattan_distance(new_state, goal_state)
        cost_diff = new_cost - current_cost
        if cost_diff <= 0 or random.random() < math.exp(-cost_diff / temperature):
            current_state = [row[:] for row in new_state]
            current_cost = new_cost
            path.append(move)
            if current_cost < best_cost:
                best_state = [row[:] for row in current_state]
                best_cost = current_cost
                best_path = path[:]
        temperature *= cooling_rate
        if temperature < min_temperature:
            break

    confidence = 1 / (1 + best_cost / max_cost) if max_cost > 0 else 0.0
    if best_state == goal_state:
        return best_path, 1.0, step_count
    return (best_path if best_path else None), confidence, step_count

def beam_search(start_state, goal_state, beam_width=2):
    queue = [(manhattan_distance(start_state, goal_state), start_state, [], 1.0)]
    visited = set()
    while queue:
        current_level = queue
        queue = []
        next_states = []
        for _, state, path, conf in current_level:
            if state == goal_state:
                return path, 1.0
            state_tuple = state_to_tuple(state)
            if state_tuple in visited:
                continue
            visited.add(state_tuple)
            for move in moves:
                new_state = move_tile(state, move)
                if new_state:
                    new_tup = state_to_tuple(new_state)
                    if new_tup not in visited:
                        new_cost = manhattan_distance(new_state, goal_state)
                        move_confidence = 1 / (1 + new_cost / (manhattan_distance(start_state, goal_state) + 10)) if (manhattan_distance(start_state, goal_state) + 10) > 0 else 0.0
                        combined_conf = (conf + move_confidence) / 2
                        next_states.append((new_cost, new_state, path + [move], combined_conf))
        next_states.sort(key=lambda x: x[3], reverse=True)
        queue = next_states[:beam_width]
    return None, 0.0

def genetic_algorithm(start_state, goal_state, population_size=100, generations=500, mutation_rate=0.1, max_path_length=50):
    start_time = time.time()
    timeout = 10
    step_count = 0

    def generate_individual():
        current = [row[:] for row in start_state]
        path = []
        visited = {state_to_tuple(current)}
        for _ in range(max_path_length):
            x, y = find_blank(current)
            possible_moves = [move for move, (dx, dy) in moves.items() if is_valid(x + dx, y + dy)]
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            next_state = move_tile(current, move)
            if next_state:
                tup = state_to_tuple(next_state)
                if tup not in visited:
                    current = next_state
                    visited.add(tup)
                    path.append(move)
                else:
                    break
        return path

    def apply_path(state, path):
        current = [row[:] for row in state]
        visited = {state_to_tuple(current)}
        valid_path = []
        for move in path:
            if move not in moves:
                break
            next_state = move_tile(current, move)
            if not next_state:
                break
            tup = state_to_tuple(next_state)
            if tup not in visited:
                current = next_state
                visited.add(tup)
                valid_path.append(move)
            else:
                break
        return current, valid_path

    def fitness(individual):
        final_state, valid_path = apply_path(start_state, individual)
        if final_state == goal_state:
            return 0
        distance = manhattan_distance(final_state, goal_state)
        correct_tiles = sum(1 for i in range(3) for j in range(3) if final_state[i][j] == goal_state[i][j])
        return distance - correct_tiles

    def tournament_selection(population, max_cost, tournament_size=5):
        tournament = random.sample(population, min(tournament_size, len(population)))
        scores = [(fitness(ind), ind) for ind in tournament]
        scores.sort()
        return scores[0][1]

    def crossover(parent1, parent2):
        if not parent1 or not parent2:
            return parent1 or parent2
        point = random.randint(0, min(len(parent1), len(parent2)) - 1)
        child = parent1[:point] + parent2[point:]
        return child[:max_path_length]

    def mutate(individual):
        mutated = individual[:]
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.choice(list(moves.keys()))
        return mutated[:max_path_length]

    max_cost = manhattan_distance(start_state, goal_state) + 10
    population = [generate_individual() for _ in range(population_size)]
    best_solution = None
    best_fitness = float('inf')
    best_confidence = 0.0

    for generation in range(generations):
        if time.time() - start_time > timeout:
            break
        step_count += population_size

        for individual in population:
            fit = fitness(individual)
            if fit == 0:
                final_state, valid_path = apply_path(start_state, individual)
                if final_state == goal_state:
                    return valid_path, 1.0, step_count
            if fit < best_fitness:
                best_fitness = fit
                best_solution = individual
                distance = manhattan_distance(apply_path(start_state, individual)[0], goal_state)
                best_confidence = 1 / (1 + distance / max_cost) if max_cost > 0 else 0.0

        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, max_cost)
            parent2 = tournament_selection(population, max_cost)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    if best_solution:
        final_state, valid_path = apply_path(start_state, best_solution)
        if final_state == goal_state:
            return valid_path, 1.0, step_count
        distance = manhattan_distance(final_state, goal_state)
        confidence = 1 / (1 + distance / max_cost) if max_cost > 0 else 0.0
        return (valid_path if valid_path else None), confidence, step_count
    return None, 0.0, step_count

def and_or_search(start_state, goal_state, max_depth=50):
    initial_belief = {state_to_tuple(start_state)}
    max_cost = manhattan_distance(start_state, goal_state) + 10

    def tuple_to_state(tup):
        return [list(row) for row in tup]

    def solve_belief(belief, depth, visited_plans, path_so_far):
        if depth <= 0:
            confidence = max(
                (1 / (1 + manhattan_distance(tuple_to_state(state), goal_state) / max_cost) for state in belief),
                default=0.0
            )
            return None, confidence

        confidences = [
            1 / (1 + manhattan_distance(tuple_to_state(state), goal_state) / max_cost) for state in belief
        ]
        belief_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        if state_to_tuple(goal_state) in belief:
            return path_so_far, 1.0

        best_plan = None
        best_confidence = belief_confidence
        best_path = None

        for move in moves:
            next_belief = set()
            move_valid = False
            for state in belief:
                list_state = tuple_to_state(state)
                new_state = move_tile(list_state, move)
                if new_state:
                    next_belief.add(state_to_tuple(new_state))
                    move_valid = True

            if not move_valid:
                continue

            plan_tuple = (frozenset(next_belief), move)
            if plan_tuple in visited_plans:
                continue
            visited_plans.add(plan_tuple)

            sub_plan, sub_confidence = solve_belief(
                next_belief, depth - 1, visited_plans, path_so_far + [move]
            )

            if sub_plan is not None:
                return sub_plan, sub_confidence
            if sub_confidence > best_confidence:
                best_confidence = sub_confidence
                best_path = path_so_far + [move]

        return best_path, best_confidence

    visited_plans = set()
    path, confidence = solve_belief(initial_belief, max_depth, visited_plans, [])

    if path:
        current = [row[:] for row in start_state]
        for move in path:
            current = move_tile(current, move)
            if current is None:
                return None, confidence
        if current == goal_state:
            return path, 1.0
    return path if path else None, confidence

def partial_observable_search(start_state, goal_state):
    """
    Môi trường quan sát một phần: ở đây chỉ dùng backtracking làm ví dụ.
    """
    result = backtracking_search(start_state, goal_state)
    if isinstance(result, tuple):
        return result
    return result, 0.0

def unknown_dynamic_search(start_state, goal_state):
    """
    Search trong môi trường động/không biết trước: làm vài bước ngẫu nhiên rồi tìm đường.
    Mô phỏng Online DFS-Agent của AIMA.
    """
    current = [row[:] for row in start_state]
    for _ in range(3):
        x, y = find_blank(current)
        possible_moves = []
        for move, (dx, dy) in moves.items():
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                possible_moves.append(move)
        if not possible_moves:
            break
        chosen_move = random.choice(possible_moves)
        current = move_tile(current, chosen_move)
    result = backtracking_search(current, goal_state)
    if isinstance(result, tuple):
        return result
    return result, 0.0

def backtracking_search(start_state, goal_state, max_depth=50):
    """
    Tìm kiếm quay lui (depth-first) đơn giản với độ sâu giới hạn.
    Tham khảo nguyên lý DFS trong AIMA CSP (hình 6.5).
    """
    visited = set()
    start_time = time.time()
    timeout = 10

    def solve(state, path):
        nonlocal visited, start_time, timeout
        if time.time() - start_time > timeout:
            return None
        if state == goal_state:
            return path
        state_tuple = state_to_tuple(state)
        if state_tuple in visited or len(path) >= max_depth:
            return None
        visited.add(state_tuple)
        for move in moves:
            new_state = move_tile(state, move)
            if new_state:
                result = solve(new_state, path + [move])
                if result is not None:
                    return result
        return None

    path = solve(start_state, [])
    if path is not None:
        return path, 1.0
    else:
        return None, 0.0

def revise(domains, xi, xj):
    revised = False
    for x_value in set(domains[xi]):
        satisfies = any(x_value != y_value for y_value in domains[xj])
        if not satisfies:
            domains[xi].remove(x_value)
            revised = True
    return revised

def forward_checking(start_state, goal_state):
    """
    Forward checking cho 8-puzzle: áp dụng AC-3 để kiểm tra nhất quán,
    sau đó dùng backtracking tìm giải.
    """
    msg, conf = ac3_algorithm(start_state)
    if conf == 0.0:
        return None, 0.0
    return backtracking_search(start_state, goal_state)

def ac3_algorithm(start_state, goal_state=None):
    """
    AC-3: kiểm tra nhất quán cung (Arc consistency) cho 8-puzzle (ràng buộc AllDifferent).
    Dựa trên pseudocode AC-3 trong AIMA (Hình 6.3).
    """
    variables = [(r, c) for r in range(3) for c in range(3)]
    domains = {}
    seen = set()
    is_valid_state = True
    for r in range(3):
        for c in range(3):
            val = start_state[r][c]
            domains[(r, c)] = {val}
            if val in seen:
                is_valid_state = False
            seen.add(val)
    if not (is_valid_state and len(seen) == 9 and all(0 <= x <= 8 for x in seen)):
        return ("AC-3: Trạng thái đầu vào không hợp lệ.", 0.0)

    queue = deque()
    for xi in variables:
        for xj in variables:
            if xi != xj:
                queue.append((xi, xj))

    while queue:
        xi, xj = queue.popleft()
        if revise(domains, xi, xj):
            if not domains[xi]:
                return (f"AC-3: Không nhất quán! Domain của ô {xi} rỗng.", 0.0)
            for xk in variables:
                if xk != xi and xk != xj:
                    queue.append((xk, xi))
    return ("AC-3: Nhất quán cung (arc-consistent) với AllDifferent.", 1.0)

def q_learning(start_state, goal_state, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.2):
    """
    Q-Learning cho bài toán 8-puzzle. start_state và goal_state là ma trận 3x3.
    Trả về đường đi (danh sách các ký tự 'L','R','U','D') từ start tới goal, và độ tin cậy (1.0 nếu tìm được).
    """
    start = [row[:] for row in start_state]
    goal = [row[:] for row in goal_state]
    goal_tuple = state_to_tuple(goal)

    actions = list(moves.keys())
    Q = {}

    for ep in range(episodes):
        state = [row[:] for row in start]
        steps = 0
        while state != goal and steps < 100:
            state_tuple = state_to_tuple(state)
            x, y = find_blank(state)
            possible_actions = []
            for a, (dx, dy) in moves.items():
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny):
                    possible_actions.append(a)
            if not possible_actions:
                break

            if random.random() < epsilon:
                action = random.choice(possible_actions)
            else:
                q_vals = [Q.get((state_tuple, a), 0.0) for a in possible_actions]
                max_q = max(q_vals)
                best_actions = [a for a, q in zip(possible_actions, q_vals) if q == max_q]
                action = random.choice(best_actions)

            new_state = move_tile(state, action)
            new_state_tuple = state_to_tuple(new_state) if new_state else state_tuple

            reward = 1 if new_state_tuple == goal_tuple else 0

            if new_state_tuple == goal_tuple:
                best_next = 0.0
            else:
                x2, y2 = find_blank(new_state)
                poss2 = []
                for a2, (dx2, dy2) in moves.items():
                    nx2, ny2 = x2 + dx2, y2 + dy2
                    if is_valid(nx2, ny2):
                        poss2.append(a2)
                if poss2:
                    best_next = max(Q.get((new_state_tuple, a2), 0.0) for a2 in poss2)
                else:
                    best_next = 0.0

            old_q = Q.get((state_tuple, action), 0.0)
            Q[(state_tuple, action)] = old_q + alpha * (reward + gamma * best_next - old_q)

            state = new_state
            steps += 1

            if state is None:
                break

    current = [row[:] for row in start]
    path = []
    visited = {state_to_tuple(current)}
    while state_to_tuple(current) != goal_tuple and len(path) < 50:
        curr_tuple = state_to_tuple(current)
        x, y = find_blank(current)
        possible_actions = []
        for a, (dx, dy) in moves.items():
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                possible_actions.append(a)
        if not possible_actions:
            break
        q_vals = [(Q.get((curr_tuple, a), 0.0), a) for a in possible_actions]
        best_action = max(q_vals, key=lambda x: x[0])[1]
        new_state = move_tile(current, best_action)
        if new_state is None:
            break
        new_tuple = state_to_tuple(new_state)
        if new_tuple in visited:
            break
        visited.add(new_tuple)
        path.append(best_action)
        current = new_state

    if state_to_tuple(current) == goal_tuple:
        return path, 1.0
    return None, 0.0

def is_solvable(flat_state):
    inv = 0
    for i in range(len(flat_state)):
        for j in range(i + 1, len(flat_state)):
            if flat_state[i] != 0 and flat_state[j] != 0 and flat_state[i] > flat_state[j]:
                inv += 1
    return inv % 2 == 0

def generate_random_state():
    while True:
        nums = list(range(9))
        random.shuffle(nums)
        if is_solvable(nums):
            return [nums[:3], nums[3:6], nums[6:]]
