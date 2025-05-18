import pygame   
import tkinter as tk
from tkinter import ttk
import os
from algorithm import *
import random
import copy

root = tk.Tk()
root.title("8-Puzzle Solver")
root.geometry("1100x700")
root.resizable(False, False)

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill="both", expand=True)

control_frame = ttk.Frame(main_frame)
control_frame.pack(side="top", fill="x", pady=5)

control_row1 = ttk.Frame(control_frame)
control_row1.pack(fill="x")

control_row2 = ttk.Frame(control_frame)
control_row2.pack(fill="x", pady=5)

control_row3 = ttk.Frame(control_frame)
control_row3.pack(fill="x", pady=5)

display_frame = ttk.PanedWindow(main_frame, orient="horizontal")
display_frame.pack(fill="both", expand=True)

canvas_frame = ttk.Frame(display_frame, width=600, height=800)
display_frame.add(canvas_frame, weight=1)

canvas = tk.Canvas(canvas_frame, width=600, height=800)
canvas.pack(fill="both", expand=True)

step_frame = ttk.Frame(display_frame, width=500, height=800)
display_frame.add(step_frame, weight=1)
step_text = tk.Text(step_frame, height=60, width=50, font=("Courier", 10))
step_scroll = ttk.Scrollbar(step_frame, orient="vertical",
command=step_text.yview)
step_text.configure(yscrollcommand=step_scroll.set)
step_scroll.pack(side="right", fill="y")
step_text.pack(side="left", fill="both", expand=True)

os.environ['SDL_WINDOWID'] = str(canvas.winfo_id())
os.environ['SDL_VIDEODRIVER'] = 'windib'
pygame.init()
screen = pygame.display.set_mode((600, 800))
pygame.display.set_caption("8-Puzzle Board")

tile_size = 40
cyan = (0, 255, 255)
white = (255, 255, 255)
blue = (0, 0, 255)
black = (0, 0, 0)
font = pygame.font.Font(None, 24)
label_font = pygame.font.Font(None, 20)

goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
initial_state = [[2, 6, 5], [0, 8, 7], [4, 3, 1]]
current_state = [row[:] for row in initial_state]
constraint_ac3_state = None  # Biến lưu trạng thái AC-3 tạo ra

solution_steps = []
current_step = 0

def draw_board(state, x_offset=50, y_offset=50):
    screen.fill(white)
    for i, row in enumerate(state):
        for j, v in enumerate(row):
            color = blue if v else white
            pygame.draw.rect(screen, color, (x_offset + j * tile_size, y_offset + i * tile_size, tile_size, tile_size))
            pygame.draw.rect(screen, black, (x_offset + j * tile_size, y_offset + i * tile_size, tile_size, tile_size), 3)
            if v:
                text = font.render(str(v), True, cyan)
                screen.blit(text, (x_offset + j * tile_size + tile_size // 3, y_offset + i * tile_size + tile_size // 4))
    pygame.display.flip()

def print_state(state):
    return "\n".join(" ".join(map(str, row)) for row in state) + "\n" + "-" * 10 + "\n"

def run_simulation():
    global solution_steps, current_step, current_state
    algorithm = algo_var.get()
    current_state = [row[:] for row in initial_state]
    result = algo_map[algorithm](current_state, goal_state)

    if isinstance(result, tuple):
        solution_steps = result[0]
    else:
        solution_steps = result

    current_step = 0
    step_text.delete(1.0, tk.END)
    step_text.insert(tk.END, "Trạng thái ban đầu:\n")
    step_text.insert(tk.END, print_state(current_state))

    if not solution_steps:
        step_text.insert(tk.END, f"Không giải được bởi {algorithm}\n")
        step_text.see(tk.END)
        draw_board(current_state)
        return

    temp_state = [row[:] for row in current_state]
    for move in solution_steps:
        temp_state = move_tile(temp_state, move)
        if temp_state is None:
            step_text.insert(tk.END, f"Lỗi: Đường đi không hợp lệ từ {algorithm}\n")
            step_text.see(tk.END)
            draw_board(current_state)
            return
    if temp_state != goal_state:
        step_text.insert(tk.END, f"Đường đi từ {algorithm} không đạt mục tiêu\n")
        step_text.see(tk.END)
        draw_board(current_state)
        return

    step_text.see(tk.END)
    draw_board(current_state)
    root.after(50, run_step)

def run_step():
    global current_step, current_state
    if current_step < len(solution_steps):
        move = solution_steps[current_step]
        try:
            current_state = move_tile(current_state, move)
            if current_state is None:
                raise ValueError(f"Nước đi không hợp lệ: {move}")
            step_text.insert(tk.END, f"Bước {current_step + 1}: {move}\n")
            step_text.insert(tk.END, print_state(current_state))
            step_text.see(tk.END)
            draw_board(current_state)
            current_step += 1
            root.after(50, run_step)
        except Exception as e:
            step_text.insert(tk.END, f"Lỗi ở bước {current_step + 1}: {str(e)}\n")
            step_text.see(tk.END)
            return
    else:
        step_text.insert(tk.END, "Đã hoàn thành!\n")
        step_text.see(tk.END)

algo_map = {
    "BFS": bfs,
    "UCS": ucs,
    "DFS": dfs,
    "IDS": ids,
    "GBFS": gbfs,
    "A*": a_star_search,
    "IDA*": ida_star,
    "Simple HC": simple_hill_climbing,
    "Steepest HC": steepest_ascent_hill_climbing,
    "Stochastic HC": stochastic_hill_climbing,
    "Simulated Annealing": simulated_annealing,
    "Beam Search": beam_search,
    "Genetic": genetic_algorithm,
    "AND-OR Search": and_or_search,
    "Partial Obs Search": partial_observable_search,
    "Unknown/Dynamic Search": unknown_dynamic_search,
    "Forward Checking": forward_checking,
    "Backtracking Search": backtracking_search,
    "AC-3 Check": ac3_algorithm,
    "Q-Learning": q_learning
}

algo_label = ttk.Label(control_row1, text="Chọn thuật toán:")
algo_label.pack(side="left", padx=5)

algo_var = tk.StringVar(value="BFS")
algo_combo = ttk.Combobox(control_row1, textvariable=algo_var, values=list(algo_map.keys()), state="readonly")
algo_combo.pack(side="left", padx=5)

run_btn = ttk.Button(control_row1, text="Run", command=run_simulation)
run_btn.pack(side="left", padx=5)

def reset():
    global current_state, solution_steps, current_step
    current_state = [row[:] for row in initial_state]
    solution_steps = []
    current_step = 0
    step_text.delete(1.0, tk.END)
    step_text.insert(tk.END, "Trạng thái ban đầu:\n")
    step_text.insert(tk.END, print_state(current_state))
    step_text.insert(tk.END, "Trạng thái mục tiêu:\n")
    step_text.insert(tk.END, print_state(goal_state))
    draw_board(current_state)

reset_btn = ttk.Button(control_row1, text="Reset", command=reset)
reset_btn.pack(side="left", padx=5)

def random_initial_state():
    global initial_state
    while True:
        random_state = generate_random_state()
        if is_solvable([cell for row in random_state for cell in row]):
            initial_state = random_state
            break
    reset()
    input_entry.delete(0, tk.END)
    input_entry.insert(0, " ".join(str(num) for row in initial_state for num in row))

random_btn = ttk.Button(control_row1, text="Random", command=random_initial_state)
random_btn.pack(side="left", padx=5)


def validate_input(text):
    if not text:
        return True
    try:
        nums = [int(x) for x in text.split()]
        if len(nums) <= 9 and all(0 <= n <= 8 for n in nums):
            return True
    except ValueError:
        return False
    return False

vcmd = (root.register(validate_input), '%P')
input_label = ttk.Label(control_row2, text="Nhập trạng thái ban đầu (0-8, cách nhau bằng khoảng trắng):")
input_label.pack(side="left", padx=5)
input_entry = ttk.Entry(control_row2, validate="key", validatecommand=vcmd, width=30)
input_entry.pack(side="left", padx=5)

def apply_custom_state():
    global current_state, initial_state
    text = input_entry.get().strip()
    if text:
        nums = [int(x) for x in text.split()]
        if len(nums) == 9 and sorted(nums) == list(range(9)):
            initial_state = [nums[:3], nums[3:6], nums[6:]]
            reset()
        else:
            step_text.delete(1.0, tk.END)
            step_text.insert(tk.END, "Trạng thái ban đầu không hợp lệ! Cần 9 số từ 0-8 không trùng lặp.\n")

apply_btn = ttk.Button(control_row2, text="Áp dụng ban đầu", command=apply_custom_state)
apply_btn.pack(side="left", padx=5)

goal_input_label = ttk.Label(control_row3, text="Nhập trạng thái đích (0-8, cách nhau bằng khoảng trắng):")
goal_input_label.pack(side="left", padx=5)
goal_input_entry = ttk.Entry(control_row3, validate="key", validatecommand=vcmd, width=30)
goal_input_entry.pack(side="left", padx=5)

def apply_custom_goal_state():
    global goal_state
    text = goal_input_entry.get().strip()
    if text:
        nums = [int(x) for x in text.split()]
        if len(nums) == 9 and sorted(nums) == list(range(9)):
            goal_state = [nums[:3], nums[3:6], nums[6:]]
            reset()
        else:
            step_text.delete(1.0, tk.END)
            step_text.insert(tk.END, "Trạng thái đích không hợp lệ! Cần 9 số từ 0-8 không trùng lặp.\n")

apply_goal_btn = ttk.Button(control_row3, text="Áp dụng đích", command=apply_custom_goal_state)
apply_goal_btn.pack(side="left", padx=5)

# Belief Window (Nhóm 4)
def open_belief_window():
    belief_window = tk.Toplevel(root)
    belief_window.title("Belief State Puzzle Solver")
    belief_window.geometry("700x400")
    belief_window.resizable(False, False)

    belief_main_frame = ttk.Frame(belief_window, padding="10")
    belief_main_frame.pack(fill="both", expand=True)

    belief_control_frame = ttk.Frame(belief_main_frame)
    belief_control_frame.pack(side="top", fill="x", pady=5)

    belief_control_row = ttk.Frame(belief_control_frame)
    belief_control_row.pack(fill="x")

    belief_algo_label = ttk.Label(belief_control_row, text="Chọn thuật toán:")
    belief_algo_label.pack(side="left", padx=5)
    belief_algo_var = tk.StringVar(value="BFS")
    belief_algo_combo = ttk.Combobox(belief_control_row, textvariable=belief_algo_var, values=list(algo_map.keys()), state="readonly")
    belief_algo_combo.pack(side="left", padx=5)

    belief_display_frame = ttk.PanedWindow(belief_main_frame, orient="horizontal")
    belief_display_frame.pack(fill="both", expand=True)

    belief_canvas_frame = ttk.Frame(belief_display_frame, width=600, height=800)
    belief_display_frame.add(belief_canvas_frame, weight=1)

    belief_canvas = tk.Canvas(belief_canvas_frame, width=600, height=800)
    belief_scroll = ttk.Scrollbar(belief_canvas_frame, orient="vertical", command=belief_canvas.yview)
    belief_canvas.configure(yscrollcommand=belief_scroll.set)
    belief_scroll.pack(side="right", fill="y")
    belief_canvas.pack(side="left", fill="both", expand=True)

    belief_inner_frame = ttk.Frame(belief_canvas)
    belief_canvas.create_window((0, 0), window=belief_inner_frame, anchor="nw")

    belief_pygame_canvas = tk.Canvas(belief_inner_frame, width=600, height=18000)
    belief_pygame_canvas.pack(fill="both", expand=True)

    belief_step_frame = ttk.Frame(belief_display_frame, width=500, height=800)
    belief_display_frame.add(belief_step_frame, weight=1)
    belief_step_text = tk.Text(belief_step_frame, height=60, width=50, font=("Courier", 10))
    belief_step_scroll = ttk.Scrollbar(belief_step_frame, orient="vertical", command=belief_step_text.yview)
    belief_step_text.configure(yscrollcommand=belief_step_scroll.set)
    belief_step_scroll.pack(side="right", fill="y")
    belief_step_text.pack(side="left", fill="both", expand=True)

    os.environ['SDL_WINDOWID'] = str(belief_pygame_canvas.winfo_id())
    os.environ['SDL_VIDEODRIVER'] = 'windib'
    pygame.init()
    belief_screen = pygame.display.set_mode((600, 18000))
    pygame.display.set_caption("Belief State Board")

    num_beliefs = 150
    fixed_numbers = [1, 2, 3]
    remaining_numbers = [0, 4, 5, 6, 7, 8]
    belief_states = []
    goal_states = []

    for _ in range(num_beliefs):
        random.shuffle(remaining_numbers)
        state = [
            fixed_numbers[:],
            remaining_numbers[:3],
            remaining_numbers[3:]
        ]
        belief_states.append(state)

    for _ in range(num_beliefs):
        random.shuffle(remaining_numbers)
        state = [
            fixed_numbers[:],
            remaining_numbers[:3],
            remaining_numbers[3:]
        ]
        goal_states.append(state)

    belief_current_states = [copy.deepcopy(state) for state in belief_states]
    belief_solution_steps = [[] for _ in range(num_beliefs)]
    belief_current_step = 0
    belief_state_history = [[copy.deepcopy(state)] for state in belief_states]
    belief_tried_moves = [[set()] for _ in range(num_beliefs)]

    def draw_state(state, x_offset, y_offset):
        for i, row in enumerate(state):
            for j, v in enumerate(row):
                color = blue if v else white
                pygame.draw.rect(belief_screen, color, (x_offset + j * tile_size, y_offset + i * tile_size, tile_size, tile_size))
                pygame.draw.rect(belief_screen, black, (x_offset + j * tile_size, y_offset + i * tile_size, tile_size, tile_size), 3)
                if v:
                    text = font.render(str(v), True, cyan)
                    belief_screen.blit(text, (x_offset + j * tile_size + tile_size // 3, y_offset + i * tile_size + tile_size // 4))

    def draw_belief_board(belief_states, goal_states, x_offset=20, y_offset=20):
        belief_screen.fill(white)
        for idx in range(len(belief_states)):
            state_y_offset = y_offset + idx * (tile_size * 3 + 30)
            belief_label = label_font.render(f"Niềm tin {idx + 1}", True, black)
            belief_screen.blit(belief_label, (x_offset, state_y_offset - 20))
            draw_state(belief_states[idx], x_offset, state_y_offset)
            goal_label = label_font.render(f"Mục tiêu {idx + 1}", True, black)
            goal_x_offset = x_offset + tile_size * 4 + 20
            belief_screen.blit(goal_label, (goal_x_offset, state_y_offset - 20))
            draw_state(goal_states[idx], goal_x_offset, state_y_offset)
        pygame.display.flip()

    def update_scroll_region():
        belief_canvas.configure(scrollregion=(0, 0, 600, num_beliefs * (tile_size * 3 + 30) + 20))

    def is_valid_belief_state(state):
        return state[0] == fixed_numbers

    def get_valid_moves(state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
                    break
        valid_moves = []
        for move, (dx, dy) in moves.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = move_tile(state, move)
                if new_state and is_valid_belief_state(new_state):
                    valid_moves.append(move)
        return valid_moves

    def run_belief_simulation():
        global belief_current_states, belief_solution_steps, belief_current_step, belief_state_history, belief_tried_moves
        algorithm = belief_algo_var.get()
        belief_current_states = [copy.deepcopy(state) for state in belief_states]
        belief_solution_steps = [[] for _ in range(num_beliefs)]
        belief_current_step = 0
        belief_state_history = [[copy.deepcopy(state)] for state in belief_states]
        belief_tried_moves = [[set()] for _ in range(num_beliefs)]
        unsolvable_beliefs = []  # Track unsolvable beliefs

        belief_step_text.delete(1.0, tk.END)
        for i in range(num_beliefs):
            belief_step_text.insert(tk.END, f"Trạng thái niềm tin {i+1}:\n")
            belief_step_text.insert(tk.END, print_state(belief_states[i]))
            belief_step_text.insert(tk.END, f"Trạng thái đích {i+1}:\n")
            belief_step_text.insert(tk.END, print_state(goal_states[i]))
        belief_step_text.see(tk.END)

        # Solve all belief states using the selected algorithm
        belief_step_text.insert(tk.END, f"Đang chạy {algorithm} cho từng niềm tin...\n")
        belief_step_text.see(tk.END)
        for i in range(num_beliefs):
            result = algo_map[algorithm](belief_states[i], goal_states[i])
            if isinstance(result, tuple):
                belief_solution_steps[i] = result[0] or []
            else:
                belief_solution_steps[i] = result or []
            if not belief_solution_steps[i]:
                belief_step_text.insert(tk.END, f"Không giải được niềm tin {i+1} bởi {algorithm}\n")
                belief_step_text.see(tk.END)
                unsolvable_beliefs.append(i + 1)
            if (i + 1) % 10 == 0 or i + 1 == num_beliefs:
                belief_step_text.insert(tk.END, f"Đã xử lý {i + 1}/{num_beliefs} niềm tin...\n")
                belief_step_text.see(tk.END)
                belief_window.update()

        # Apply all moves immediately for each belief state that was solvable
        belief_step_text.insert(tk.END, f"Hoàn tất {algorithm}. Áp dụng tất cả bước đi...\n")
        belief_step_text.see(tk.END)

        for i in range(num_beliefs):
            if not belief_solution_steps[i]:
                continue
            belief_state_history[i] = [copy.deepcopy(belief_current_states[i])]
            belief_tried_moves[i] = [set()]
            for step, move in enumerate(belief_solution_steps[i]):
                try:
                    new_state = move_tile(belief_current_states[i], move)
                    if new_state is None or not is_valid_belief_state(new_state):
                        valid_moves = get_valid_moves(belief_current_states[i])
                        valid_moves = [m for m in valid_moves if m not in belief_tried_moves[i][-1]]
                        if valid_moves:
                            new_move = random.choice(valid_moves)
                            new_state = move_tile(belief_current_states[i], new_move)
                            if new_state and is_valid_belief_state(new_state):
                                belief_solution_steps[i][step] = new_move
                                belief_tried_moves[i][-1].add(new_move)
                                belief_step_text.insert(tk.END, f"Bước {step + 1} (Niềm tin {i+1}): {new_move}\n")
                                belief_step_text.insert(tk.END, print_state(new_state))
                            else:
                                belief_tried_moves[i][-1].add(new_move)
                                belief_step_text.insert(tk.END, f"Bước {step + 1} (Niềm tin {i+1}): Thử nước đi {new_move} không hợp lệ, giữ nguyên trạng thái\n")
                                belief_step_text.insert(tk.END, print_state(belief_current_states[i]))
                                continue
                        else:
                            belief_step_text.insert(tk.END, f"Bước {step + 1} (Niềm tin {i+1}): Không tìm thấy nước đi hợp lệ, giữ nguyên trạng thái\n")
                            belief_step_text.insert(tk.END, print_state(belief_current_states[i]))
                            continue
                    belief_current_states[i] = new_state
                    belief_state_history[i].append(copy.deepcopy(new_state))

                    belief_tried_moves[i].append(set())
                    belief_step_text.insert(tk.END, f"Bước {step + 1} (Niềm tin {i+1}): {belief_solution_steps[i][step]}\n")
                    belief_step_text.insert(tk.END, print_state(belief_current_states[i]))
                except Exception as e:
                    belief_step_text.insert(tk.END, f"Lỗi ở bước {step + 1} (Niềm tin {i+1}): {str(e)}\n")
                    belief_step_text.see(tk.END)
                    return
            belief_step_text.see(tk.END)
            belief_window.update()

        # Check success and update display
        successful_beliefs = sum(1 for i in range(num_beliefs) if belief_current_states[i] == goal_states[i])
        if successful_beliefs >= 2:
            belief_step_text.insert(tk.END, f"Đã hoàn thành! {successful_beliefs}/{num_beliefs} niềm tin đạt trạng thái đích.\n")
        else:
            belief_step_text.insert(tk.END, f"Không thành công! Chỉ {successful_beliefs}/{num_beliefs} niềm tin đạt trạng thái đích.\n")
        if unsolvable_beliefs:
            belief_step_text.insert(tk.END, f"Các niềm tin không giải được: {', '.join(map(str, unsolvable_beliefs))}.\n")
        belief_step_text.see(tk.END)

        draw_belief_board(belief_current_states, goal_states)
        update_scroll_region()

    def reset_belief():
        global belief_current_states, belief_solution_steps, belief_current_step, belief_state_history, belief_tried_moves
        belief_current_states = [copy.deepcopy(state) for state in belief_states]
        belief_solution_steps = [[] for _ in range(num_beliefs)]
        belief_current_step = 0
        belief_state_history = [[copy.deepcopy(state)] for state in belief_states]
        belief_tried_moves = [[set()] for _ in range(num_beliefs)]
        belief_step_text.delete(1.0, tk.END)
        for i in range(num_beliefs):
            belief_step_text.insert(tk.END, f"Trạng thái niềm tin {i+1}:\n")
            belief_step_text.insert(tk.END, print_state(belief_states[i]))
            belief_step_text.insert(tk.END, f"Trạng thái đích {i+1}:\n")
            belief_step_text.insert(tk.END, print_state(goal_states[i]))
        draw_belief_board(belief_current_states, goal_states)
        update_scroll_region()
        belief_step_text.see(tk.END)

    run_belief_btn = ttk.Button(belief_control_row, text="Run Belief Simulation", command=run_belief_simulation)
    run_belief_btn.pack(side="left", padx=5)

    reset_belief_btn = ttk.Button(belief_control_row, text="Reset Belief", command=reset_belief)
    reset_belief_btn.pack(side="left", padx=5)

    draw_belief_board(belief_current_states, goal_states)
    update_scroll_region()

    def belief_main_loop():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                belief_window.destroy()
                pygame.quit()
        belief_window.after(10, belief_main_loop)

    belief_window.after(10, belief_main_loop)

# Constraint Window (Nhóm 5)
def open_constraint_window():
    constraint_window = tk.Toplevel(root)
    constraint_window.title("Constraint Puzzle Solver")
    constraint_window.geometry("600x400")
    constraint_window.resizable(False, False)

    constraint_main_frame = ttk.Frame(constraint_window, padding="10")
    constraint_main_frame.pack(fill="both", expand=True)

    # Khung chọn thuật toán và nút
    constraint_control_frame = ttk.Frame(constraint_main_frame)
    constraint_control_frame.pack(side="top", fill="x", pady=5)
    constraint_control_row = ttk.Frame(constraint_control_frame)
    constraint_control_row.pack(fill="x")
    ttk.Label(constraint_control_row, text="Chọn thuật toán:").pack(side="left", padx=5)
    constraint_algo_var = tk.StringVar(value="AC-3")
    constraint_algo_combo = ttk.Combobox(
        constraint_control_row, textvariable=constraint_algo_var,
        values=["AC-3", "Backtracking", "Forward Checking"], state="readonly"
    )
    constraint_algo_combo.pack(side="left", padx=5)

    # Khung hiển thị kết quả
    constraint_step_text = tk.Text(constraint_main_frame, height=15, width=70, font=("Courier", 10))
    constraint_step_scroll = ttk.Scrollbar(constraint_main_frame, orient="vertical", command=constraint_step_text.yview)
    constraint_step_text.configure(yscrollcommand=constraint_step_scroll.set)
    constraint_step_scroll.pack(side="right", fill="y")
    constraint_step_text.pack(side="left", fill="both", expand=True)

    def run_constraint_simulation():
        global initial_state, constraint_ac3_state
        algo = constraint_algo_var.get()
        constraint_step_text.delete(1.0, tk.END)

        if algo == "AC-3":
            from algorithm import generate_random_state, is_solvable, ac3_algorithm
            constraint_ac3_state = None  # Xóa trạng thái cũ khi bắt đầu
            constraint_step_text.insert(tk.END, "Random trạng thái ban đầu hợp lệ...\n")
            # Tạo ngẫu nhiên trạng thái hợp lệ
            while True:
                random_state = generate_random_state()
                if is_solvable([cell for row in random_state for cell in row]):
                    temp_initial = random_state
                    break
            # In trạng thái và chạy AC-3
            constraint_step_text.insert(tk.END, "Trạng thái ban đầu (ngẫu nhiên):\n")
            constraint_step_text.insert(tk.END, print_state(temp_initial))
            constraint_step_text.insert(tk.END, "Trạng thái mục tiêu:\n")
            constraint_step_text.insert(tk.END, print_state(goal_state))
            msg, conf = ac3_algorithm(temp_initial)
            constraint_step_text.insert(tk.END, f"{msg}\n")
            if conf == 0.0:
                constraint_step_text.insert(tk.END, "AC-3 thất bại. Không tiếp tục giải.\n")
                constraint_step_text.see(tk.END)
                return
            # Nếu AC-3 thành công, lưu trạng thái
            constraint_ac3_state = temp_initial
            constraint_step_text.see(tk.END)
            return

        elif algo == "Backtracking":
            from algorithm import backtracking_search, move_tile
            # Chọn trạng thái sử dụng: ưu tiên constraint_ac3_state nếu tồn tại
            if constraint_ac3_state is not None:
                used_state = constraint_ac3_state
            else:
                used_state = initial_state
            constraint_step_text.insert(tk.END, "Trạng thái ban đầu:\n")
            constraint_step_text.insert(tk.END, print_state(used_state))
            constraint_step_text.insert(tk.END, "Trạng thái mục tiêu:\n")
            constraint_step_text.insert(tk.END, print_state(goal_state))
            path, conf = backtracking_search(used_state, goal_state)
            if not path:
                constraint_step_text.insert(tk.END, "Không tìm thấy giải pháp với Backtracking.\n")
            else:
                temp_state = [row[:] for row in used_state]
                for step, move in enumerate(path):
                    temp_state = move_tile(temp_state, move)
                    constraint_step_text.insert(tk.END, f"Bước {step+1}: {move}\n")
                    constraint_step_text.insert(tk.END, print_state(temp_state))
            constraint_step_text.see(tk.END)

        elif algo == "Forward Checking":
            from algorithm import forward_checking, move_tile
            # Tương tự: chọn trạng thái ưu tiên từ AC-3 nếu có
            if constraint_ac3_state is not None:
                used_state = constraint_ac3_state
            else:
                used_state = initial_state
            constraint_step_text.insert(tk.END, "Trạng thái ban đầu:\n")
            constraint_step_text.insert(tk.END, print_state(used_state))
            constraint_step_text.insert(tk.END, "Trạng thái mục tiêu:\n")
            constraint_step_text.insert(tk.END, print_state(goal_state))
            path, conf = forward_checking(used_state, goal_state)
            if not path:
                constraint_step_text.insert(tk.END, "Không tìm thấy giải pháp với Forward Checking.\n")
            else:
                temp_state = [row[:] for row in used_state]
                for step, move in enumerate(path):
                    temp_state = move_tile(temp_state, move)
                    constraint_step_text.insert(tk.END, f"Bước {step+1}: {move}\n")
                    constraint_step_text.insert(tk.END, print_state(temp_state))
            constraint_step_text.see(tk.END)

    def reset_constraint():
        # Xóa và in lại trạng thái ban đầu/mục tiêu
        constraint_step_text.delete(1.0, tk.END)
        used_state = constraint_ac3_state if constraint_ac3_state is not None else initial_state
        constraint_step_text.insert(tk.END, "Trạng thái ban đầu:\n")
        constraint_step_text.insert(tk.END, print_state(used_state))
        constraint_step_text.insert(tk.END, "Trạng thái mục tiêu:\n")
        constraint_step_text.insert(tk.END, print_state(goal_state))

    ttk.Button(constraint_control_row, text="Run", command=run_constraint_simulation).pack(side="left", padx=5)
    ttk.Button(constraint_control_row, text="Reset", command=reset_constraint).pack(side="left", padx=5)

# Nút mở Belief (Niềm Tin - Nhóm 4) đã có:
belief_btn = ttk.Button(control_row1, text="Niềm Tin", command=open_belief_window)
belief_btn.pack(side="left", padx=5)

# Nút mở Constraint Window (Nhóm 5)
constraint_btn = ttk.Button(control_row1, text="Ràng buộc", command=open_constraint_window)
constraint_btn.pack(side="left", padx=5)

draw_board(current_state)

def main_loop():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            root.quit()
    root.after(10, main_loop)

root.after(10, main_loop)
root.mainloop()
pygame.quit()
