import numpy as np
from environments.lrtdp import lrtdp
from baseline.helper_functions import *
from baseline.feedback_methods import *

REGULARIZATION_THRESHOLD = 0.0001
REGULARIZATION_PENALTY = 0.001
def apply_regularization(grid, state, actions):
    actions_list = list(actions)
    q_values = [grid.agent_q_values[(state, a)] for a in actions_list]
    for i, q1 in enumerate(q_values):
        for j, q2 in enumerate(q_values):
            if i != j and abs(q1 - q2) < REGULARIZATION_THRESHOLD:
                grid.agent_q_values[(state, actions_list[i])] += REGULARIZATION_PENALTY
                grid.agent_q_values[(state, actions_list[j])] += REGULARIZATION_PENALTY


def softmax_probability(grid, state, actions, beta, t, unknown_theta):
    numerator = 0
    prob_sum = 0
    state_probs = []
    if isinstance(state, str):
        state = eval(state)
    denominator = np.sum([np.exp(-beta * grid.agent_q_values[(state, a_prime)]) for a_prime in actions])
    for a in actions:
        numerator = np.exp(-beta * grid.agent_q_values[(state, a)])
        prob = numerator / denominator
        state_probs.append(prob)
        prob_sum += prob * np.log(prob / prior_distribution(t, len(unknown_theta)))

    return prob_sum, state_probs

def prior_distribution(theta, total_theta_values):
    prior = np.ones_like(theta) / total_theta_values
    return prior

def expected_kl_divergence(grid, unknown_theta, all_beta, oracle_policy, agent_policy, num_feedback, output_dir):
    max_ekl = [float('-inf'), float('-inf')]
    total = 0.0
    log_likelihoods = []
    for t_idx, t in enumerate(unknown_theta):
        feedback_methods = get_feedback_methods()
        for i, method in enumerate(feedback_methods):
            beta = all_beta[i]
            if method=='correction':
                filename = f"{output_dir}callibration_feedback/{t_idx}_Correction.txt"
                if not os.path.exists(filename):
                    print(f"File {filename} does not exist.")
                    return {}
                state_action_labels = {}
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    for line in lines[1:]:
                        parts = line.strip().rsplit(', ', 2)
                        if len(parts) == 3:
                            state, action, label = parts
                            state_action_labels[(state, action)] = label

            elif method=='annotated_correction':
                filename = f"{output_dir}callibration_feedback/{t_idx}_Annotated Correction.txt"
                if not os.path.exists(filename):
                    print(f"File {filename} does not exist.")
                    return {}
                state_action_labels = {}
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    for line in lines[1:]:
                        parts = line.strip().rsplit(', ', 2)
                        if len(parts) == 3:
                            state, action, label = parts
                            state_action_labels[(state, action)] = label

            elif method=='approval':
                filename = f"{output_dir}callibration_feedback/{t_idx}_Approval.txt"
                if not os.path.exists(filename):
                    print(f"File {filename} does not exist.")
                    return {}
                state_action_labels = {}
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    for line in lines[1:]:
                        parts = line.strip().rsplit(', ', 2)
                        if len(parts) == 3:
                            state, action, label = parts
                            state_action_labels[(state, action)] = label
            elif method=='annotated_approval':
                filename = f"{output_dir}callibration_feedback/{t_idx}_Annotated Approval.txt"
                if not os.path.exists(filename):
                    print(f"File {filename} does not exist.")
                    return {}
                state_action_labels = {}
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    for line in lines[1:]:
                        parts = line.strip().rsplit(', ', 2)
                        if len(parts) == 3:
                            state, action, label = parts
                            state_action_labels[(state, action)] = label
            elif method=='dam':
                filename = f"{output_dir}callibration_feedback/{t_idx}_Demo-Action Mismatch.txt"
                if not os.path.exists(filename):
                    print(f"File {filename} does not exist.")
                    return {}
                state_action_labels = {}
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    for line in lines[1:]:
                        parts = line.strip().rsplit(', ', 2)
                        if len(parts) == 3:
                            state, action, label = parts
                            state_action_labels[(state, action)] = label
            elif method=='rank':
                filename = f"{output_dir}callibration_feedback/{t_idx}_Ranking.txt"
                if not os.path.exists(filename):
                    print(f"File {filename} does not exist.")
                    return {}
                state_action_labels = {}
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    for line in lines[1:]:
                        parts = line.strip().rsplit(', ', 2)
                        if len(parts) == 3:
                            state, action, label = parts
                            state_action_labels[(state, action)] = label
        state_action_pairs = get_preferred_state_action_pairs(state_action_labels)

        grid.calibration_reward = t
        grid.get_all_reward = True
        _ = lrtdp(grid, is_oracle=False)
        theta_prob_sum = 0.0
        for state, actions in state_action_pairs.items():
            if isinstance(state, str):
                state = eval(state)
            prob_sum, state_probs = softmax_probability(grid, state, actions, beta, t, unknown_theta)
            theta_prob_sum += prob_sum
            for sa_prob in state_probs:
                sa_prob = max(sa_prob, 1e-10)
                log_likelihood += np.log(sa_prob)
        log_likelihoods.append(log_likelihood)
        total += theta_prob_sum
        ekl = total / len(unknown_theta)
        if (ekl > max_ekl).all():
            max_ekl = ekl
            likelihoods = np.exp(log_likelihoods - np.max(log_likelihoods))
            posterior = likelihoods / np.sum(likelihoods)
            theta_val = unknown_theta[np.argmax(posterior)]
    return theta_val


def active_feedback_selection(grid, beta, oracle_policy, agent_policy, num_feedback, unknown_theta, output_dir):
    theta_val = expected_kl_divergence(grid, unknown_theta, beta, oracle_policy, agent_policy, num_feedback, output_dir)
    return theta_val
