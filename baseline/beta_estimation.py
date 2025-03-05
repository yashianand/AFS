from baseline.helper_functions import *
from baseline.feedback_methods import *
from environments.lrtdp import lrtdp
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

def mle_objective(grid, beta, state_action_labels):
    beta_est_val = 0
    state_action_pairs = get_preferred_state_action_pairs(state_action_labels)
    for state, actions in state_action_pairs.items():
        if isinstance(state, str):
            state = eval(state)
        all_actions_in_state = [action_val for (state_, action_val) in grid.agent_q_values.keys() if state_ == state]
        normalization_term = logsumexp([-beta * grid.agent_q_values[(state, a)] for a in all_actions_in_state])
        for a in actions:
            beta_est_val += -beta * grid.agent_q_values[(state, a)] - normalization_term
    return -beta_est_val

def fit_beta(grid, state_action_labels, beta):
    result = minimize(lambda beta: mle_objective(grid, beta[0], state_action_labels), [beta], bounds=[(0, None)])
    beta = result.x[0]
    return beta

def get_oracle_feedback_and_fit_beta(grid, oracle_policy, agent_policy, beta, num_feedback, c_idx, output_dir):
    feedback_methods = get_feedback_methods()
    all_sa_pairs = get_all_sa_pairs(grid)
    for i, method in enumerate(feedback_methods):
        state_action_labels = None
        if method=='correction':
            state_action_labels = get_correction(grid, oracle_policy, agent_policy, num_feedback, all_sa_pairs)
        elif method=='approval':
            state_action_labels = get_approval(grid, oracle_policy, agent_policy, num_feedback, all_sa_pairs)
        elif method=='annotated_correction':
            state_action_labels = get_annotated_correction(grid, oracle_policy, agent_policy, num_feedback, all_sa_pairs)
        elif method=='annotated_approval':
            state_action_labels = get_annotated_approval(grid, oracle_policy, agent_policy, num_feedback, all_sa_pairs)
        elif method=='dam':
            state_action_labels = get_demo_action_mismatch(grid, oracle_policy, agent_policy, num_feedback, all_sa_pairs)
        elif method=='rank':
            state_action_labels = get_rank(grid, num_feedback, all_sa_pairs)
        if state_action_labels is not None:
            filename = f"{output_dir}new_setup_ri_callibration_feedback/{c_idx}_{method}.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    f.write("{}, {}, {}\n".format('state', 'action', 'label'))
            with open(filename, 'a') as file:
                for key, value in state_action_labels.items():
                    file.write(f"{key[0]}, {key[1]}, {value}\n")
            file.close()
        beta[i] = fit_beta(grid, state_action_labels, beta[i])
    return beta

def estimate_beta(grid, num_feedback, calibration_rewards, output_dir):
    feedback_beta_cap = np.zeros(len(get_feedback_methods()))
    for c_idx, reward in enumerate(calibration_rewards):
        grid.calibration_reward = reward
        calibration_oracle_policy = lrtdp(grid, is_oracle=True)
        oracle_q_values = grid.oracle_q_values
        calibration_agent_policy = lrtdp(grid, is_oracle=False)
        grid.oracle_q_values = oracle_q_values
        feedback_beta_cap = get_oracle_feedback_and_fit_beta(grid, calibration_oracle_policy, calibration_agent_policy, feedback_beta_cap, num_feedback, c_idx, output_dir)
    return feedback_beta_cap
