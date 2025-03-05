from environments.helper_functions import *

def get_ann_corr_feedback(grid, critical_states, oracle_policy, initial_agent_policy):
    all_states = grid.all_states
    all_sa_pairs = get_all_sa_pairs(grid)
    for sa_pair in all_sa_pairs:
        state, action = sa_pair[0], sa_pair[1]
        state = tuple(state)
        action = int(action)
        if (state, action) not in grid.learned_reward_cache:
            grid.learned_reward_cache[(state, action)] = 0
    for cs_id in critical_states:
        c_state = tuple(all_states[cs_id])
        agent_action = int(initial_agent_policy[c_state])
        if is_safe_action(grid, c_state, agent_action) == False:
            action_label = evaluate_action(grid, c_state, agent_action)
            safe_action = int(oracle_policy[c_state])
            for a in range(grid.num_actions):
                if a != safe_action:
                    grid.learned_reward_cache[(c_state, a)] = action_label

def get_corr_feedback(grid, critical_states, oracle_policy, initial_agent_policy):
    all_states = grid.getStateFactorRep()
    all_sa_pairs = get_all_sa_pairs(grid)
    for sa_pair in all_sa_pairs:
        state, action = sa_pair[0], sa_pair[1]
        state = tuple(state)
        action = int(action)
        if (state, action) not in grid.learned_reward_cache:
            grid.learned_reward_cache[(state, action)] = 0
    for cs_id in critical_states:
        c_state = tuple(all_states[cs_id])
        agent_action = int(initial_agent_policy[c_state])
        if is_safe_action(grid, c_state, agent_action) == False:
            safe_action = int(oracle_policy[c_state])
            for a in range(grid.num_actions):
                if a != safe_action:
                    grid.learned_reward_cache[(c_state, a)] = 2
