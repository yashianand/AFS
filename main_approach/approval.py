import numpy as np
from environments.helper_functions import *

def get_random_sa_pairs(grid, critical_states):
    states = grid.getStateFactorRep()
    cs = []
    for cs_id in critical_states:
        c_state = tuple(states[cs_id])
        cs.append(c_state)
    state_action_pairs = []
    num_states = len(critical_states)
    if grid.domain=='vase' or grid.domain=='outdoor':
        random_pair_idx = np.random.choice(num_states*4, num_states, replace=False)
        for i in range(len(random_pair_idx)):
            state_id = random_pair_idx[i] // 4
            state = tuple(cs[state_id])
            action = int(random_pair_idx[i] % 4)
            state_action_pairs.append((state, action))
    else:
        all_valid_csa_pairs = []
        for c_state in cs:
            all_actions = grid.get_actions(c_state)
            for action in all_actions:
                all_valid_csa_pairs.append([c_state, action])
        random_sa_pairs = np.random.choice(len(all_valid_csa_pairs), num_states, replace=False)
        for sa in random_sa_pairs:
            random_state = all_valid_csa_pairs[sa][0]
            random_action = all_valid_csa_pairs[sa][1]
            state_action_pairs.append((random_state, random_action))
    return state_action_pairs

def get_app_feedback(grid, critical_states, oracle_policy, initial_agent_policy):
    all_sa_pairs = get_all_sa_pairs(grid)
    for sa_pair in all_sa_pairs:
        state, action = tuple(sa_pair[0]), int(sa_pair[1])
        if (state, action) not in grid.learned_reward_cache:
            grid.learned_reward_cache[(state, action)] = 0
    random_sa_pairs = get_random_sa_pairs(grid, critical_states)
    for (state, action) in random_sa_pairs:
        if is_safe_action(grid, state, action) == True:
            grid.learned_reward_cache[(state, action)] = 0
        elif is_safe_action(grid, state, action) == False:
            grid.learned_reward_cache[(state, action)] = 2

def get_ann_app_feedback(grid, critical_states, oracle_policy, initial_agent_policy):
    all_sa_pairs = get_all_sa_pairs(grid)
    for sa_pair in all_sa_pairs:
        state, action = tuple(sa_pair[0]), int(sa_pair[1])
        if (state, action) not in grid.learned_reward_cache:
            grid.learned_reward_cache[(state, action)] = 0
    random_sa_pairs = get_random_sa_pairs(grid, critical_states)
    for (state, action) in random_sa_pairs:
        if is_safe_action(grid, state, action) == True:
            grid.learned_reward_cache[(state, action)] = 0
        elif is_safe_action(grid, state, action) == False:
            action_label = evaluate_action(grid, state, action)
            grid.learned_reward_cache[(state, action)] = action_label
