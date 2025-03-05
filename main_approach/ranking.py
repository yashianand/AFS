import numpy as np
from environments.helper_functions import get_all_sa_pairs, is_safe_action

def get_random_ranking_queries(grid, critical_states, SEED=42):
    all_states = grid.getStateFactorRep()
    state_action_pairs = []
    for sa in critical_states:
        state = tuple(all_states[sa])
        all_actions = grid.get_actions(state)
        random_actions = np.random.choice(all_actions, 2, replace=False)
        action_1, action_2 = random_actions[0], random_actions[1]
        state_action_pairs.append((state, action_1, action_2))
    return state_action_pairs

def map_to_labels(grid, ranks):
    for state, action_ranks in ranks.items():
        selected_a = action_ranks[0]
        unselected_a = action_ranks[1]
        grid.learned_reward_cache[(state, selected_a)] = 0
        grid.learned_reward_cache[(state, unselected_a)] = 2

def get_rank_feedback(grid, critical_states, oracle_policy, initial_agent_policy):
    ranks = {}
    all_sa_pairs = get_all_sa_pairs(grid)
    for sa_pair in all_sa_pairs:
        state, action = tuple(sa_pair[0]), int(sa_pair[1])
        if (state, action) not in grid.learned_reward_cache:
            grid.learned_reward_cache[(state, action)] = 0
    random_queries = get_random_ranking_queries(grid, critical_states)
    for (state, a1, a2) in random_queries:
        # rank(a1) > rank(a2)
        if is_safe_action(grid, state, a1) == True and is_safe_action(grid, state, a2) == False:
            ranks[state] = [a1, a2]
        # rank(a1) < rank(a2)
        elif is_safe_action(grid, state, a1) == False and is_safe_action(grid, state, a2) == True:
            ranks[state] = [a2, a1]
        # a1, a2 equally good or bad
        else:
            selected_action = np.random.choice([a1, a2], 1)[0]
            unselected_action = a1 if selected_action == a2 else a2
            ranks[state] = [selected_action, unselected_action]
    map_to_labels(grid, ranks)
