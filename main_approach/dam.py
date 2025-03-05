from environments.helper_functions import *

def get_dam_feedback(grid, critical_states, oracle_policy, initial_agent_policy):
    all_states = grid.getStateFactorRep()
    all_sa_pairs = get_all_sa_pairs(grid)
    for sa_pair in all_sa_pairs:
        state, action = tuple(sa_pair[0]), int(sa_pair[1])
        if (state, action) not in grid.learned_reward_cache:
            grid.learned_reward_cache[(state, action)] = 0
    for cs_id in critical_states:
        c_state = tuple(all_states[cs_id])
        agent_action = int(initial_agent_policy[c_state])
        oracle_action = int(oracle_policy[c_state])
        if agent_action!=oracle_action:
            grid.learned_reward_cache[(c_state, agent_action)] = 2
