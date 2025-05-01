import heapq
from collections import deque
import math
import random
import time
import numpy as np

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
    visited = set()
    max_cost = manhattan_distance(start_state, goal_state) + 10
    while queue:
        state, path = queue.popleft()
        if state == goal_state:
            return path, 1.0
        visited.add(state_to_tuple(state))
        current_cost = manhattan_distance(state, goal_state)
        confidence = 1 / (1 + current_cost / max_cost) if max_cost > 0 else 0.0
        for move in moves:
            new_state = move_tile(state, move)
            if new_state and state_to_tuple(new_state) not in visited:
                queue.append((new_state, path + [move]))
    return None, 0.0

def ucs(start_state, goal_state):
    pq = [(0, start_state, [])]
    visited = set()
    max_cost = manhattan_distance(start_state, goal_state) + 10
    while pq:
        cost, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path, 1.0
        visited.add(state_to_tuple(state))
        current_cost = manhattan_distance(state, goal_state)
        confidence = 1 / (1 + current_cost / max_cost) if max_cost > 0 else 0.0
        for move in moves:
            new_state = move_tile(state, move)
            if new_state and state_to_tuple(new_state) not in visited:
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
            if new_state and state_to_tuple(new_state) not in visited:
                stack.append((new_state, path + [move], depth + 1))
    return None

def ids(start_state, goal_state):
    def dls(state, depth, path, visited, max_cost):
        if state == goal_state:
            return path, 1.0
        if depth == 0:
            return None, 0.0
        visited.add(state_to_tuple(state))
        current_cost = manhattan_distance(state, goal_state)
        confidence = 1 / (1 + current_cost / max_cost) if max_cost > 0 else 0.0
        for move in moves:
            new_state = move_tile(state, move)
            if new_state and state_to_tuple(new_state) not in visited:
                result, conf = dls(new_state, depth - 1, path + [move], visited, max_cost)
                if result:
                    return result, conf
        return None, confidence
    
    max_cost = manhattan_distance(start_state, goal_state) + 10
    depth = 0
    while True:
        visited = set()
        result, confidence = dls(start_state, depth, [], visited, max_cost)
        if result:
            return result, confidence
        depth += 1
        if depth > 100:
            return None, 0.0

def gbfs(start_state, goal_state):
    pq = [(manhattan_distance(start_state, goal_state), start_state, [])]
    visited = set()
    max_cost = manhattan_distance(start_state, goal_state) + 10
    while pq:
        _, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path, 1.0
        visited.add(state_to_tuple(state))
        current_cost = manhattan_distance(state, goal_state)
        confidence = 1 / (1 + current_cost / max_cost) if max_cost > 0 else 0.0
        move_confidences = []
        for move in moves:
            new_state = move_tile(state, move)
            if new_state and state_to_tuple(new_state) not in visited:
                move_cost = manhattan_distance(new_state, goal_state)
                move_confidence = 1 / (1 + move_cost / max_cost) if max_cost > 0 else 0.0
                move_confidences.append((move_confidence, new_state, move))
        move_confidences.sort(reverse=True)
        for _, new_state, move in move_confidences:
            heapq.heappush(pq, (manhattan_distance(new_state, goal_state), new_state, path + [move]))
    return None, 0.0

def a_star_search(start_state, goal_state):
    pq = [(manhattan_distance(start_state, goal_state), 0, start_state, [])]
    visited = set()
    max_cost = manhattan_distance(start_state, goal_state) + 10
    while pq:
        _, cost, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path, 1.0
        visited.add(state_to_tuple(state))
        current_cost = manhattan_distance(state, goal_state)
        confidence = 1 / (1 + current_cost / max_cost) if max_cost > 0 else 0.0
        for move in moves:
            new_state = move_tile(state, move)
            if new_state and state_to_tuple(new_state) not in visited:
                new_cost = cost + 1
                priority = new_cost + manhattan_distance(new_state, goal_state)
                heapq.heappush(pq, (priority, new_cost, new_state, path + [move]))
    return None, 0.0

def ida_star(start_state, goal_state):
    bound = manhattan_distance(start_state, goal_state)
    max_cost = manhattan_distance(start_state, goal_state) + 10
    
    def search(state, g, bound, path):
        f = g + manhattan_distance(state, goal_state)
        if f > bound:
            return f, None, 0.0
        if state == goal_state:
            return f, path, 1.0
        min_cost = math.inf
        confidence = 1 / (1 + manhattan_distance(state, goal_state) / max_cost) if max_cost > 0 else 0.0
        for move in moves:
            new_state = move_tile(state, move)
            if new_state:
                result, new_path, new_conf = search(new_state, g + 1, bound, path + [move])
                if new_path:
                    return result, new_path, new_conf
                min_cost = min(min_cost, result)
        return min_cost, None, confidence

    while True:
        result, path, confidence = search(start_state, 0, bound, [])
        if path:
            return path, confidence
        if result == math.inf:
            return None, 0.0
        bound = result

def simple_hill_climbing(start_state, goal_state):
    current_state = [row[:] for row in start_state]
    path = []
    max_cost = manhattan_distance(start_state, goal_state) + 10
    while True:
        best_neighbor = None
        best_distance = manhattan_distance(current_state, goal_state)
        best_move = None
        confidence = 1 / (1 + best_distance / max_cost) if max_cost > 0 else 0.0
        move_confidences = []
        for move in moves:
            new_state = move_tile(current_state, move)
            if new_state:
                new_distance = manhattan_distance(new_state, goal_state)
                move_confidence = 1 / (1 + new_distance / max_cost) if max_cost > 0 else 0.0
                move_confidences.append((move_confidence, new_state, move))
        if not move_confidences:
            return path, confidence
        best_confidence, best_neighbor, best_move = max(move_confidences)
        if manhattan_distance(best_neighbor, goal_state) >= manhattan_distance(current_state, goal_state):
            return path, confidence
        current_state = best_neighbor
        path.append(best_move)

def steepest_ascent_hill_climbing(start_state, goal_state):
    current_state = [row[:] for row in start_state]
    path = []
    max_cost = manhattan_distance(start_state, goal_state) + 10
    while True:
        neighbors = []
        confidence = 1 / (1 + manhattan_distance(current_state, goal_state) / max_cost) if max_cost > 0 else 0.0
        for move in moves:
            new_state = move_tile(current_state, move)
            if new_state:
                move_confidence = 1 / (1 + manhattan_distance(new_state, goal_state) / max_cost) if max_cost > 0 else 0.0
                neighbors.append((move_confidence, new_state, move))
        if not neighbors:
            return path, confidence
        best_confidence, best_state, best_move = max(neighbors)
        if manhattan_distance(best_state, goal_state) >= manhattan_distance(current_state, goal_state):
            return path, confidence
        current_state = best_state
        path.append(best_move)

def stochastic_hill_climbing(start_state, goal_state):
    current_state = [row[:] for row in start_state]
    path = []
    max_cost = manhattan_distance(start_state, goal_state) + 10
    while True:
        neighbors = []
        confidence = 1 / (1 + manhattan_distance(current_state, goal_state) / max_cost) if max_cost > 0 else 0.0
        for move in moves:
            new_state = move_tile(current_state, move)
            if new_state:
                move_confidence = 1 / (1 + manhattan_distance(new_state, goal_state) / max_cost) if max_cost > 0 else 0.0
                neighbors.append((move_confidence, new_state, move))
        if not neighbors:
            return path, confidence
        better_neighbors = [(conf, state, move) for conf, state, move in neighbors 
                           if manhattan_distance(state, goal_state) < manhattan_distance(current_state, goal_state)]
        if better_neighbors:
            confidences = [conf for conf, _, _ in better_neighbors]
            total_conf = sum(confidences)
            probabilities = [conf / total_conf if total_conf > 0 else 1/len(better_neighbors) 
                            for conf in confidences]
            chosen_state, chosen_move = random.choices(
                [(state, move) for _, state, move in better_neighbors], 
                weights=probabilities, k=1)[0]
            current_state = chosen_state
            path.append(chosen_move)
        else:
            return path, confidence

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
    timeout = 10
    
    for iteration in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        step_count += 1
        if current_state == goal_state:
            return path, 1.0, step_count
        x, y = find_blank(current_state)
        possible_moves = [move for move, (dx, dy) in moves.items() if is_valid(x + dx, y + dy)]
        if not possible_moves:
            break
        move_confidences = []
        for move in possible_moves:
            new_state = move_tile(current_state, move)
            new_cost = manhattan_distance(new_state, goal_state)
            move_confidence = 1 / (1 + new_cost / max_cost) if max_cost > 0 else 0.0
            move_confidences.append((move_confidence, move, new_state))
        confidences = [conf for conf, _, _ in move_confidences]
        total_conf = sum(confidences)
        probabilities = [conf / total_conf if total_conf > 0 else 1/len(possible_moves) 
                        for conf in confidences]
        _, move, new_state = random.choices(move_confidences, weights=probabilities, k=1)[0]
        new_cost = manhattan_distance(new_state, goal_state)
        cost_diff = new_cost - current_cost
        confidence = 1 / (1 + new_cost / max_cost) if max_cost > 0 else 0.0
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
    return best_path if best_path else None, confidence, step_count

def beam_search(start_state, goal_state, beam_width=2):
    queue = [(manhattan_distance(start_state, goal_state), start_state, [], 1.0)]
    visited = set()
    max_cost = manhattan_distance(start_state, goal_state) + 10
    while queue:
        current_level = queue
        queue = []
        next_states = []
        for _, state, path, conf in current_level:
            if state == goal_state:
                return path, 1.0
            if state_to_tuple(state) in visited:
                continue
            visited.add(state_to_tuple(state))
            for move in moves:
                new_state = move_tile(state, move)
                if new_state and state_to_tuple(new_state) not in visited:
                    new_cost = manhattan_distance(new_state, goal_state)
                    move_confidence = 1 / (1 + new_cost / max_cost) if max_cost > 0 else 0.0
                    combined_conf = (conf + move_confidence) / 2
                    next_states.append((new_cost, new_state, path + [move], combined_conf))
        next_states.sort(key=lambda x: x[3], reverse=True)
        queue = next_states[:beam_width]
    return None, 0.0

def nondeterministic_search(start_state, goal_state):
    pq = [(manhattan_distance(start_state, goal_state), 0, start_state, [], 1.0)]
    visited = set()
    max_cost = manhattan_distance(start_state, goal_state) + 10
    while pq:
        _, cost, state, path, conf = heapq.heappop(pq)
        if state == goal_state:
            return path, 1.0
        state_tuple = state_to_tuple(state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        x, y = find_blank(state)
        possible_moves = [move for move, (dx, dy) in moves.items() if is_valid(x + dx, y + dy)]
        next_states = []
        for move in possible_moves:
            new_state = move_tile(state, move)
            if new_state and state_to_tuple(new_state) not in visited:
                new_cost = manhattan_distance(new_state, goal_state)
                move_confidence = 1 / (1 + new_cost / max_cost) if max_cost > 0 else 0.0
                combined_conf = (conf + move_confidence) / 2
                next_states.append((new_cost, new_state, move, combined_conf))
        next_states.sort(key=lambda x: x[3], reverse=True)
        selected_moves = next_states[:2] if len(next_states) > 2 else next_states
        random.shuffle(selected_moves)
        for _, new_state, move, move_conf in selected_moves:
            new_cost = cost + 1
            priority = new_cost + manhattan_distance(new_state, goal_state)
            heapq.heappush(pq, (priority, new_cost, new_state, path + [move], move_conf))
    return None, 0.0

def genetic_algorithm(start_state, goal_state, population_size=100, generations=500, mutation_rate=0.1, max_path_length=50):
    start_time = time.time()
    timeout = 10
    step_count = 0
    
    def generate_individual():
        current = [row[:] for row in start_state]
        path = []
        visited = set()
        visited.add(state_to_tuple(current))
        for _ in range(max_path_length):
            x, y = find_blank(current)
            possible_moves = [move for move, (dx, dy) in moves.items() if is_valid(x + dx, y + dy)]
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            next_state = move_tile(current, move)
            if next_state and state_to_tuple(next_state) not in visited:
                current = next_state
                visited.add(state_to_tuple(current))
                path.append(move)
        return path

    def apply_path(state, path):
        current = [row[:] for row in state]
        valid_path = []
        visited = set()
        visited.add(state_to_tuple(current))
        for move in path:
            if not isinstance(move, str) or move not in moves:
                break
            next_state = move_tile(current, move)
            if next_state and state_to_tuple(next_state) not in visited:
                current = next_state
                visited.add(state_to_tuple(current))
                valid_path.append(move)
            else:
                break
        return current, valid_path

    def fitness(individual):
        final_state, valid_path = apply_path(start_state, individual)
        distance = manhattan_distance(final_state, goal_state)
        correct_tiles = sum(1 for i in range(3) for j in range(3) if final_state[i][j] == goal_state[i][j])
        if final_state == goal_state:
            return 0
        return distance - correct_tiles

    def calculate_confidence(individual, max_cost):
        final_state, valid_path = apply_path(start_state, individual)
        distance = manhattan_distance(final_state, goal_state)
        return 1 / (1 + distance / max_cost) if max_cost > 0 else 0.0

    def tournament_selection(population, max_cost, tournament_size=5):
        tournament = random.sample(population, min(tournament_size, len(population)))
        confidences = [calculate_confidence(ind, max_cost) for ind in tournament]
        fitnesses = [fitness(ind) for ind in tournament]
        scores = [(f + (1 - c) * max_cost) for f, c in zip(fitnesses, confidences)]
        return tournament[np.argmin(scores)]

    def crossover(parent1, parent2):
        if not parent1 or not parent2:
            return parent1 if parent1 else parent2
        point = random.randint(0, min(len(parent1), len(parent2)) - 1)
        child = parent1[:point] + parent2[point:]
        return [move for move in child if isinstance(move, str) and move in moves][:max_path_length]

    def mutate(individual):
        if not individual:
            return individual
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
        
        fitness_values = [fitness(individual) for individual in population]
        confidence_values = [calculate_confidence(ind, max_cost) for ind in population]
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else float('inf')
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        
        for individual in population:
            fit = fitness(individual)
            if fit == 0:
                final_state, valid_path = apply_path(start_state, individual)
                if final_state == goal_state:
                    return valid_path, 1.0, step_count
            if fit < best_fitness:
                best_fitness = fit
                best_solution = individual
                best_confidence = calculate_confidence(individual, max_cost)
        
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
        return valid_path if valid_path else None, best_confidence, step_count
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


def trial_and_error_search(start_state, goal_state, max_steps=5000):
    """
    Tìm kiếm bằng cách thử và sai (Random Walk).
    Di chuyển ngẫu nhiên cho đến khi tìm thấy đích hoặc hết số bước/thời gian.
    """
    current_state = [row[:] for row in start_state]
    path = []
    steps = 0
    start_time = time.time()
    timeout = 10  # Giới hạn thời gian 10 giây

    max_cost = manhattan_distance(start_state, goal_state) + 10 # Để tính confidence

    while steps < max_steps and time.time() - start_time < timeout:
        if current_state == goal_state:
            return path, 1.0 # Tìm thấy đích, confidence = 1.0

        x, y = find_blank(current_state)
        possible_moves = []
        for move, (dx, dy) in moves.items():
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                possible_moves.append(move)

        if not possible_moves:
            break # Bị kẹt, không có nước đi nào

        # Chọn một nước đi ngẫu nhiên từ các nước đi hợp lệ
        chosen_move = random.choice(possible_moves)
        new_state = move_tile(current_state, chosen_move)

        # Di chuyển đến trạng thái mới
        current_state = new_state
        path.append(chosen_move)
        steps += 1

    # Nếu vòng lặp kết thúc mà không tìm thấy đích
    final_dist = manhattan_distance(current_state, goal_state)
    confidence = 1 / (1 + final_dist / max_cost) if max_cost > 0 else 0.0
    # Trả về đường đi đã thực hiện và confidence dựa trên khoảng cách cuối cùng
    return path, confidence

def backtracking_search(start_state, goal_state, max_depth=50):
    """
    Thuật toán tìm kiếm quay lui (Backtracking Search).
    Giống DFS nhưng thường được cài đặt đệ quy và có thể giới hạn độ sâu.
    """
    visited = set()
    start_time = time.time()
    timeout = 10  # Giới hạn thời gian 10 giây

    def solve(current_state, current_path):
        nonlocal visited, start_time, timeout # Cho phép thay đổi biến bên ngoài

        # Kiểm tra timeout
        if time.time() - start_time > timeout:
            return None # Hết thời gian

        # Điều kiện dừng: tìm thấy đích
        if current_state == goal_state:
            return current_path

        state_tuple = state_to_tuple(current_state)

        # Điều kiện dừng: đã thăm hoặc vượt quá độ sâu tối đa
        if state_tuple in visited or len(current_path) >= max_depth:
            return None

        # Đánh dấu đã thăm
        visited.add(state_tuple)

        # Thử các nước đi có thể
        # Sử dụng list(moves.keys()) để có thứ tự cố định hoặc random.sample để ngẫu nhiên
        for move in moves.keys():
            new_state = move_tile(current_state, move)
            if new_state:
                # Gọi đệ quy
                result = solve(new_state, current_path + [move])
                # Nếu tìm thấy lời giải từ nhánh này, trả về ngay
                if result is not None:
                    return result

        # Nếu không tìm thấy lời giải từ trạng thái này (cần quay lui)
        # Với visited toàn cục, không cần xóa state_tuple khỏi visited khi quay lui
        return None

    # Bắt đầu tìm kiếm
    path = solve(start_state, [])

    # Trả về kết quả và confidence
    if path is not None:
        return path, 1.0 # Tìm thấy lời giải
    else:
        # Không tìm thấy lời giải (do hết thời gian, hết độ sâu, hoặc không có lời giải)
        return None, 0.0 # Confidence = 0.0 khi thất bại
    

def revise(domains, xi, xj):
    """
    Hàm helper cho AC-3. Kiểm tra và loại bỏ các giá trị trong domain của Xi
    mà không có giá trị tương ứng trong domain của Xj thỏa mãn ràng buộc.
    Ràng buộc ở đây là Xi != Xj.
    Trả về True nếu domain của Xi bị thay đổi, False nếu không.
    """
    revised = False
    domain_xi = list(domains[xi]) # Tạo bản sao để duyệt và xóa an toàn

    for x_value in domain_xi:
        # Kiểm tra xem có tồn tại y_value trong domain của Xj sao cho x_value != y_value không
        satisfies = False
        for y_value in domains[xj]:
            if x_value != y_value:
                satisfies = True
                break # Chỉ cần tìm thấy một y_value là đủ

        # Nếu không tìm thấy y_value nào thỏa mãn, loại bỏ x_value khỏi domain của Xi
        if not satisfies:
            domains[xi].remove(x_value)
            revised = True

    return revised

def ac3_algorithm(start_state, goal_state=None): # goal_state không thực sự dùng ở đây
    """
    Thực hiện thuật toán AC-3 để kiểm tra tính nhất quán cung (Arc Consistency)
    cho trạng thái bắt đầu của bài toán 8-puzzle.
    Ràng buộc chính là AllDifferent (mỗi ô phải có giá trị khác nhau).
    """
    variables = []
    for r in range(3):
        for c in range(3):
            variables.append((r, c)) # Biến là vị trí (row, col)

    # Khởi tạo domain: Ban đầu, mỗi biến chỉ có thể nhận giá trị tại vị trí đó trong start_state.
    # Nếu muốn kiểm tra tổng quát hơn, có thể khởi tạo domain là {0, ..., 8} cho tất cả,
    # nhưng với mục đích demo trên trạng thái đã cho, dùng giá trị cụ thể sẽ nhanh hơn.
    domains = {}
    is_valid_start_state = True
    seen_values = set()
    flat_state = []
    for r in range(3):
        for c in range(3):
            val = start_state[r][c]
            flat_state.append(val)
            domains[(r, c)] = {val} # Domain ban đầu chỉ chứa giá trị hiện tại
            if val in seen_values:
                is_valid_start_state = False # Phát hiện trùng lặp ngay từ đầu
            seen_values.add(val)

    # Kiểm tra cơ bản xem trạng thái đầu vào có hợp lệ không (đủ 9 số 0-8)
    if not (is_valid_start_state and len(seen_values) == 9 and all(0 <= x <= 8 for x in seen_values)):
         # Trả về tuple (thông báo, confidence)
         return ("AC-3: Trạng thái đầu vào không hợp lệ (không đủ số 0-8 hoặc có số trùng lặp).", 0.0)

    # Tạo hàng đợi chứa tất cả các cung (arcs)
    # Cung tồn tại giữa mọi cặp biến khác nhau do ràng buộc AllDifferent
    queue = deque()
    for i in range(len(variables)):
        for j in range(len(variables)):
            if i != j:
                queue.append((variables[i], variables[j]))

    # Xử lý hàng đợi
    while queue:
        (xi, xj) = queue.popleft()

        # Gọi hàm revise
        if revise(domains, xi, xj):
            # Nếu domain của Xi bị rỗng -> không nhất quán
            if not domains[xi]:
                # Trả về tuple (thông báo, confidence)
                return (f"AC-3: Phát hiện không nhất quán! Domain của ô {xi} trở nên rỗng.", 0.0)

            # Thêm lại các cung liên quan đến Xi vào hàng đợi (trừ cung ngược lại xj, xi)
            # Vì các lân cận của Xi cần kiểm tra lại với Xi đã bị thay đổi domain
            for xk in variables:
                if xk != xi and xk != xj: # Chỉ xét các hàng xóm k khác i và j
                    # Kiểm tra xem có ràng buộc giữa xk và xi không (luôn có trong AllDifferent)
                    queue.append((xk, xi)) # Thêm cung (xk, xi)

    # Nếu thuật toán kết thúc mà không tìm thấy sự không nhất quán
    # Trả về tuple (thông báo, confidence)
    return ("AC-3: Trạng thái đã cho là nhất quán cung (arc-consistent) với ràng buộc AllDifferent.", 1.0)