from environments.helper_functions import *

def get_annotated_correction(grid, oracle_policy, agent_policy, num_feedback, all_sa_pairs):
    agent_demos = get_demonstration(grid, num_trials=num_feedback, policy=agent_policy, is_oracle=False)
    state_action_labels = {}
    for sa_pair in all_sa_pairs:
        state, action = sa_pair[0], sa_pair[1]
        if (state, action) not in state_action_labels:
            state_action_labels[(state, action)] = 0
        if state in agent_demos:
            agent_action = agent_demos[state]
            if is_safe_action(grid, state, agent_action) == False:
                action_label = evaluate_action(grid, state, agent_action)
                safe_action = int(oracle_policy[state])
                for a in range(grid.num_actions):
                    if a != safe_action:
                        state_action_labels[(state, a)] = action_label
    return state_action_labels

def get_annotated_approval(grid, oracle_policy, agent_policy, num_feedback, all_sa_pairs):
    max_allowed_feedbacks = (grid.num_states-grid.num_walls)*grid.num_actions
    if num_feedback > max_allowed_feedbacks:
        num_feedback = max_allowed_feedbacks
    random_sa_pairs = get_random_state_action_pairs(grid, num_feedback)
    state_action_labels = {}
    for sa_pair in all_sa_pairs:
        state, action = sa_pair[0], sa_pair[1]
        if (state, action) not in state_action_labels:
            state_action_labels[(state, action)] = 0
        if (state, action) in random_sa_pairs:
            if is_safe_action(grid, state, action) == False:
                action_label = evaluate_action(grid, state, action)
                state_action_labels[(state, action)] = action_label
    return state_action_labels

def get_demo_action_mismatch(grid, oracle_policy, agent_policy, num_feedback, all_sa_pairs):
    get_demonstration(grid, num_trials=num_feedback, policy=oracle_policy)
    state_action_labels = {}
    for sa_pair in all_sa_pairs:
        state, action = sa_pair[0], sa_pair[1]
        if (state, action) not in state_action_labels:
            state_action_labels[(state, action)] = 0
        agent_action = int(agent_policy[state])
        if state in grid.oracle_demos:
            oracle_action = int(grid.oracle_demos[state])
            if agent_action != oracle_action:
                state_action_labels[(state, agent_action)] = 2
    return state_action_labels


def get_correction(grid, oracle_policy, agent_policy, num_feedback, all_sa_pairs):
    agent_demos = get_demonstration(grid, num_trials=num_feedback, policy=agent_policy, is_oracle=False)
    state_action_labels = {}
    for sa_pair in all_sa_pairs:
        state, action = sa_pair[0], sa_pair[1]
        if (state, action) not in state_action_labels:
            state_action_labels[(state, action)] = 0
        if state in agent_demos:
            agent_action = agent_demos[state]
            if is_safe_action(grid, state, agent_action) == False:
                safe_action = int(oracle_policy[state])
                for a in range(grid.num_actions):
                    if a != safe_action:
                        state_action_labels[(state, a)] = 2
    return state_action_labels

def get_approval(grid, oracle_policy, agent_policy, num_feedback, all_sa_pairs):
    max_allowed_feedbacks = (grid.num_states-grid.num_walls)*grid.num_actions
    if num_feedback > max_allowed_feedbacks:
        num_feedback = max_allowed_feedbacks
    random_sa_pairs = get_random_state_action_pairs(grid, num_feedback)
    state_action_labels = {}
    for sa_pair in all_sa_pairs:
        state, action = sa_pair[0], sa_pair[1]
        if (state, action) not in state_action_labels:
            state_action_labels[(state, action)] = 0

        if (state, action) in random_sa_pairs:
            if is_safe_action(grid, state, action) == False:
                state_action_labels[(state, action)] = 2
    return state_action_labels

def get_rank(grid, num_feedback, all_sa_pairs, SEED=42):
    max_allowed_feedbacks = (grid.num_states-grid.num_walls)*grid.num_actions
    if num_feedback > max_allowed_feedbacks:
        num_feedback = max_allowed_feedbacks
    random_queries = get_random_ranking_queries(grid, num_feedback, SEED)
    state_action_labels = {}
    ranks = {}
    for sa_pair in all_sa_pairs:
        state, action = sa_pair[0], sa_pair[1]
        if (state, action) not in state_action_labels:
            state_action_labels[(state, action)] = 0
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
    state_action_labels = map_to_labels(state_action_labels, ranks)
    return state_action_labels

def map_to_labels(state_action_labels, ranks):
    for state, action_ranks in ranks.items():
        selected_a = action_ranks[0]
        unselected_a = action_ranks[1]
        state_action_labels[(state, selected_a)] = 0
        state_action_labels[(state, unselected_a)] = 2
    return state_action_labels
