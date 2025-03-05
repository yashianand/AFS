import numpy as np

def valueIteration(grid, is_oracle=True, gamma=0.99, epsilon=0.01, for_q_value=False):
    all_actions = range(grid.num_actions)
    v_new = {s: 0 for s in grid.all_states}
    pi_new = {s: 0 for s in grid.all_states}
    q_value = {s: [] for s in grid.all_states}

    while True:
        v = v_new.copy()
        delta = 0
        for state in grid.all_states:
            (x,y), _ = state
            value = []
            if grid.domain=='bp':
                all_actions = grid.get_actions(state)
            for action in range(all_actions):
                successors, succ_probabilities = grid.get_successors(state, action)
                value.append(sum(succ_probabilities[i] * v[successors[i]] for i in range(len(successors))))
                if grid.is_goal(state) == True:
                    value[action] = grid.get_reward(state, action, is_oracle)
                    continue
                if (tuple(state), action) in grid.agent_reward_cache:
                    reward = grid.agent_reward_cache[(tuple(state), action)]
                else:
                    reward = grid.get_reward(state, action, is_oracle)
                value[action] = reward + gamma * value[action]
            q_value[state] = value
            v_new[state] = max(value)
            pi_new[state] = max(q_value, key=lambda k: q_value[state][k])
            delta = max(delta, abs((v_new[state]) - (v[state])))
        if delta < epsilon:
            if for_q_value==False:
                return v_new, pi_new
            else:
                return v_new, pi_new, q_value
