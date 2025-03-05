from main_approach.helper_functions import *
from main_approach.reward_pred_model import pred_cs_and_get_qcap, update_reward_model
import numpy as np

def choose_feedback(feedback_types, t, epsilon, f_G, f_N, f_costs, baseline=None):
    fb_val = np.zeros(len(feedback_types))
    fb_probability = get_feedback_probs()
    for i, f in enumerate(feedback_types):
        probability = fb_probability[i]
        cost = f_costs[i]
        f_gain = f_G[i]
        if baseline!=None:
            f_gain = 1
            if baseline=='cost-sensitive':
                probability = 1
            elif baseline=='most-probable':
                cost = 1
        fb_val[i] = probability * (1/ (f_gain*cost))
        if baseline!=None:
            last_term = 0
        else:
            last_term = np.sqrt(np.log(t)/(f_N[i] + epsilon))
        fb_val[i] += last_term
    feedback_ch = np.argmax(fb_val)
    return feedback_ch, fb_val

def learn(grid, oracle_policy, initial_agent_policy, labels, output_dir, n_clusters, tot_budget, baseline, epsilon=0.001):
    grid.reset()
    trained_model = None
    feedback_types = ['correction', 'approval', 'dam', 'annotated_correction', 'annotated_approval', 'ranking']
    f_N = np.zeros(len(feedback_types))
    t = 1
    budget = tot_budget
    f_costs = get_feedback_costs()
    f_G = np.zeros(len(feedback_types)) + epsilon
    method_queried = ''
    method_received = ''
    all_fg, all_budget = [], []
    p_cap, q_cap = [], []
    all_fb_val = []
    critical_states = []
    m = get_agent_initial_sa_labels(grid)

    while budget > 0:
        critical_states = sample_critical_states(grid, t, p_cap, q_cap, m, critical_states, labels, n_clusters)
        all_budget.append(budget)
        all_fg.append(list(f_G))

        # 1. agent chooses a feedback method
        feedback_ch, fb_val = choose_feedback(feedback_types, t, epsilon, f_G, f_N, f_costs, baseline=baseline)
        all_fb_val.append(list(fb_val))
        if feedback_types[feedback_ch] in ['correction', 'approval', 'dam', 'ranking']:
            method_queried += feedback_types[feedback_ch][0]+'-'
        elif feedback_types[feedback_ch] == 'annotated_correction':
            method_queried += 'ac-'
        else:
            method_queried += 'aa-'
        # 2. oracle provides feedback
        if get_chosen_feedback(feedback_ch, baseline=baseline)!=None:
            print('chosen feedback: ', feedback_types[feedback_ch])
            collect_oracle_feedback(grid, critical_states, feedback_ch, oracle_policy, initial_agent_policy)
            if feedback_types[feedback_ch] in ['correction', 'approval', 'dam', 'ranking']:
                method_received += feedback_types[feedback_ch][0]+'-'
            elif feedback_types[feedback_ch] == 'annotated_correction':
                method_received += 'ac-'
            else:
                method_received += 'aa-'

            # 3. agent approximates oracle's action labels (safe/unsafe)
            if p_cap!=[]:
                m = q_cap
            p_cap = approximate_p_cap(grid)

            # 4. train model -> agent learns from the approximation
            trained_model = update_reward_model(grid)

            # update variables
            f_N[feedback_ch] += 1
            q_cap = pred_cs_and_get_qcap(grid, critical_states, trained_model)
            f_G = update_feedback_gain(p_cap, q_cap, feedback_ch, f_G, critical_states)

        budget -= f_costs[feedback_ch]
        t += 1
    learned_reward_to_file(grid, output_dir+"learned_reward_smartQ/", tot_budget, method_queried, is_main_approach=True)
    learned_reward_to_file(grid, output_dir+"cs_pred_reward/", tot_budget, method_received, cs=True, is_main_approach=True)
    write_val_to_file(all_fg, all_budget, output_dir+'fg_budget_vals/', tot_budget)
    write_val_to_file(all_fb_val, all_budget, output_dir+'fb_vals/', tot_budget)


    _, updated_agent_policy = update_agent_policy(grid, trained_model)
    return trained_model
