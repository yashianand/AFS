import numpy as np

def greedy_action(grid, state, v):
    all_actions = grid.get_actions(state)
    q_values = {action: qValue(grid, state, action, v) for action in all_actions}
    best_action = min(q_values, key=lambda k: q_values[k])
    for action, q_val in q_values.items():
        if grid.is_oracle == True:
            grid.oracle_q_values[(state, action)] = q_val
        else:
            grid.agent_q_values[(state, action)] = q_val
    return best_action, q_values[best_action]

def qValue(grid, state, action, v):
    if grid.is_goal(state):
        state_val =  grid.get_reward(state, action, is_oracle=grid.is_oracle, is_lrtdp=True)
        return state_val
    successors, succ_probabilities = grid.get_successors(state, action)
    state_val = sum(succ_probabilities[i] * v[successors[i]] for i in range(len(successors)))
    if (state, action) in grid.agent_reward_cache:
        state_val += grid.agent_reward_cache[(state, action)]
    else:
        state_val += grid.get_reward(state, action, is_oracle=grid.is_oracle, is_lrtdp=True)
    return state_val

def update(grid, state, v, pi):
    action, q_val = greedy_action(grid, state, v)
    pi[state] = action
    v[state] = q_val

def residual(grid, state, v):
    _, q_val = greedy_action(grid, state, v)
    return abs(v[state] - q_val)

def pickNextState(grid, state, action):
    successors, succ_probabilities = grid.get_successors(state, action)
    next_state_id = np.random.choice(len(successors), p=succ_probabilities)
    return successors[next_state_id]

def checkSolved(grid, state, epsilon, SOLVED, v, pi):
    open_set = {state}
    closed_set = set()
    while open_set:
        state = open_set.pop()
        closed_set.add(state)
        if residual(grid, state, v) <= epsilon:
            probable_successors = set()
            action, _ = greedy_action(grid, state, v)
            successors, succ_probabilities = grid.get_successors(state, action)
            for i in range(len(successors)):
                if succ_probabilities[i] > 0:
                    probable_successors.add(successors[i])
            open_set.update(probable_successors - SOLVED - closed_set - open_set)
        else:
            while closed_set:
                state = closed_set.pop()
                update(grid, state, v, pi)
            return False
    SOLVED.update(closed_set)
    return True

def lrtdp_trial(grid, state, epsilon, SOLVED, v, pi):
    visited = []
    while state not in SOLVED:
        visited.append(state)
        if grid.is_goal(state):
            break
        update(grid, state, v, pi)
        action = int(pi[state])  # use policy directly
        state = pickNextState(grid, state, action)
    while visited:
        state = visited.pop()
        if not checkSolved(grid, state, epsilon, SOLVED, v, pi):
            break

def printEnvironment(grid, vals,  policy=False):
    res = ""
    for r in range(grid.rows):
        res += "|"
        for c in range(grid.cols):
            if policy:
                val = ["Left", "Up", "Right", "Down"][int(vals[r][c])]
            else:
                val = str([vals[r][c]])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)

def lrtdp(grid, epsilon=0.001, max_trials=250, is_oracle=True, use_cache=False, val=0):
    grid.is_oracle = is_oracle
    trial = 1
    nS = grid.num_states
    v = {s: 0 for s in grid.all_states}
    pi = {s: 0 for s in grid.all_states}
    SOLVED = set()
    if use_cache==False:
        grid.reset()
    state = grid.start_state
    while state not in SOLVED and trial <= max_trials:
        lrtdp_trial(grid, state, epsilon, SOLVED, v, pi)
        trial += 1
    return pi
