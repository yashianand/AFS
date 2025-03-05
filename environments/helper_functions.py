import numpy as np
import os
from pathlib import Path
import itertools

def simulate_trajectory(grid, policy):
    trajectory = []
    trial_reward = 0
    terminal= False
    grid.reset(full_grid=False)
    action = int(policy[grid.state])
    trajectory.append([tuple(grid.state), action])
    observation, reward, _, terminal = grid.step(action)
    action = int(policy[observation])
    trajectory.append([tuple(observation), action])
    trial_reward += reward
    while not terminal:
        observation, reward, _, terminal = grid.step(action)
        action = int(policy[observation])
        trajectory.append([tuple(observation), action])
        trial_reward += reward
    return trial_reward, trajectory

def get_pi_reward(grid, policy, num_trials=100):
    rewards = []
    for i in range(num_trials):
        trial_reward, _ = simulate_trajectory(grid, policy)
        rewards.append(trial_reward)
    return np.mean(rewards), np.std(rewards)

def get_demonstration(grid, num_trials, policy, is_oracle=True):
    agent_demos = {}
    for _ in range(num_trials):
        _, trajectory = simulate_trajectory(grid, policy)
        for step in range(len(trajectory)):
            state = trajectory[step][0]
            action = trajectory[step][1]
            if is_oracle:
                if state not in grid.oracle_demos:
                    grid.oracle_demos[state] = action
            elif not is_oracle:
                if state not in agent_demos:
                    agent_demos[state] = action
    return agent_demos

def evaluate_action(grid, state, action):
    if grid.domain=='vase':
        if state[1] == True and state[2] == False: # severe NSE
            if action in [0,1,2,3]:
                return 2
        elif state[1] == True and state[2] == True: # mild NSE
            if action in [0,1,2,3]:
                return 1
        else: # no NSE
            if action in [0,1,2,3]:
                return 0
    elif grid.domain=='outdoor':
        if state[1]==False and state[2] == True: # severe NSE
            if action in [0,1,2,3]:
                return 2
        elif state[1] == False and state[2] == False: # mild NSE
            if action in [0,1,2,3]:
                return 1
        else: # no NSE
            if action in [0,1,2,3]:
                return 0
    elif grid.domain=='bp':
        if state[1] == True and state[2] == False and state[3]==True:
            return 2
        elif state[1] == True and state[2] == False and state[3] == False:
            return 1
        else:
            return 0

def is_safe_action(grid, state, action):
    if grid.domain=='vase':
        if (state[1] == True and state[2] == False) or (state[1] == True and state[2] == True):
            if action in [0,1,2,3]:
                return False
    elif grid.domain=='outdoor':
        if (state[1]==False and state[2] == True) or (state[1] == False and state[2] == False):
            if action in [0,1,2,3]:
                return False
        else:
            return True
    else:
        if state[0]==grid.box_loc and state[1]==False:
            if action==5:
                return False
        elif state[1] == True and state[2] == False:
            if action in [0,1,2,3,4,5]:
                return False

    return True

def get_random_state_action_pairs(grid, curr_budget):
    all_states = grid.getStateFactorRep()
    if grid.domain=='bp':
        state_action_pairs = []
        all_valid_state_action_pairs = get_all_sa_pairs(grid)
        random_sa_pairs = np.random.choice(len(all_valid_state_action_pairs), int(curr_budget), replace=False)
        for sa in random_sa_pairs:
            random_state = all_valid_state_action_pairs[sa][0]
            random_action = all_valid_state_action_pairs[sa][1]
            state_action_pairs.append((random_state, random_action))
    else:
        num_states_not_wall = grid.num_states - grid.num_walls
        total_choices = num_states_not_wall * grid.num_actions
        indices = np.random.choice(total_choices, int(curr_budget), replace=False)
        state_indices = indices // grid.num_actions
        actions = indices % grid.num_actions
        state_action_pairs = [(all_states[state_idx], action) for state_idx, action in zip(state_indices, actions)]
    return state_action_pairs


def get_all_sa_pairs(grid):
    state_action_pairs = []
    all_actions = None
    if grid.domain=='vase' or grid.domain=='outdoor':
        all_actions = range(grid.num_actions)
    for state in grid.all_states:
        if grid.domain=='bp':
            all_actions = grid.get_actions(state)
        for action in all_actions:
            state_action_pairs.append([state, action])
    return state_action_pairs

def get_nse(grid):
    severe_nse, mild_nse = np.zeros((grid.rows, grid.cols)), np.zeros((grid.rows, grid.cols))
    for state in grid.getStateFactorRep():
        if grid.domain=='vase':
            if state[1] == True and state[2]==False:
                severe_nse[state[0][0]][state[0][1]] = 1
            elif state[1] == True and state[2] == True:
                mild_nse[state[0][0]][state[0][1]] = 1
        elif grid.domain=='outdoor':
            if state[1]==False and state[2]==True:
                severe_nse[state[0][0]][state[0][1]] = 1
            elif state[1] == False and state[2] == False:
                mild_nse[state[0][0]][state[0][1]] = 1
        elif grid.domain=='bp':
            if state[1]==True and state[2]==False and state[3]==True:
                severe_nse[state[0][0]][state[0][1]] = 1
            elif state[1]==True and state[2]==False and state[3]==False:
                mild_nse[state[0][0]][state[0][1]] = 1
    return severe_nse, mild_nse

def learned_reward_to_file(grid, output_dir, curr_budget, method,  cs=False, is_main_approach=False):
    all_states = grid.getStateFactorRep()
    if is_main_approach:
        filename = Path(output_dir+str(curr_budget)+'.csv')
    else:
        filename = Path(output_dir+str(curr_budget)+str(method)+'.csv')
    filename.parent.mkdir(parents=True, exist_ok=True)
    dictionary = grid.learned_reward_cache
    if cs==True:
        dictionary = grid.critical_state_preds

    with open(filename, 'w') as f:
        if os.stat(filename).st_size == 0:
            f.write("iteration,(state,action),penalty\n")
        for s in all_states:
            s = tuple(s)
            for a in range(grid.num_actions):
                if (s, a) in dictionary:
                    f.write('{},{},{}\n'.format(curr_budget, (s, a), dictionary[(s, a)]))
        f.close()

    filename1 = Path(output_dir+'feedback_methods.csv')
    filename1.parent.mkdir(parents=True, exist_ok=True)
    with open(filename1, 'a') as f1:
        if os.stat(filename1).st_size == 0:
            f1.write("iteration,methods\n")
        f1.write('{},{}\n'.format(curr_budget, method))
        f1.close()

def get_all_sa_pairs(grid):
    state_action_pairs = []
    all_actions = None
    if grid.domain=='vase' or grid.domain=='outdoor':
        all_actions = range(grid.num_actions)
    for state in grid.all_states:
        if grid.domain=='bp':
            all_actions = grid.get_actions(state)
        for action in all_actions:
            state_action_pairs.append([state, action])
    return state_action_pairs

def get_all_s_a1_a2_pairs(grid):
    all_states = grid.getStateFactorRep()
    state_action_combinations = []
    for state in all_states:
        all_actions = grid.get_actions(state)
        action_combinations = itertools.combinations(all_actions, 2)
        for action1, action2 in action_combinations:
            state_action_combinations.append((state, action1, action2))
    return state_action_combinations

def get_random_ranking_queries(grid, curr_budget, SEED):
    all_state_action_combinations = get_all_s_a1_a2_pairs(grid)
    budget = min(curr_budget, len(all_state_action_combinations))
    random_indices = np.random.choice(len(all_state_action_combinations), budget, replace=False)
    return [all_state_action_combinations[idx] for idx in random_indices]

def get_state_action_successors(grid):
    state_action_pairs = []
    if grid.domain=='vase' or grid.domain=='outdoor':
        all_actions = range(grid.num_actions)
    for state in grid.getStateFactorRep():
        if grid.domain=='bp':
            all_actions = grid.get_actions(state)
        for action in all_actions:
            if grid.get_successors(state, action)!=[]:
                state_action_pairs.append((tuple(state), action))
    return state_action_pairs
