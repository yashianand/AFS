from main_approach.reward_pred_model import get_state_action_successors
from environments.nse_analysis import *
from environments.helper_functions import *
from main_approach.helper_functions import update_agent_policy
import numpy as np, os

def pred_test_grid(test_map_names, model, output_dir, filename, iterations, is_safety_grid):
    to_write = ['iterations', 'map_name', 'accuracy', 'fpRate', 'fnRate', 'avg_severe_nse', 'severe_nse_sd', 'avg_mild_nse', 'mild_nse_sd']
    for test_map_name in test_map_names:
        test_grid = readMDPfile(filename, test_map_name, is_safety_grid=is_safety_grid, is_oracle=False, is_lrtdp=True)
        test_grid.reset()
        all_test_state_action_pairs = get_state_action_successors(test_grid)
        f1_calculation_data = []
        state_pred, test_grid_agent_policy = update_agent_policy(test_grid, model)
        pred_fn = output_dir+'/test_grid_predictions_'+str(test_map_name)+'.csv'
        with open(pred_fn, 'a') as f:
            if os.stat(pred_fn).st_size == 0:
                f.write("{},{},{},{}\n".format('iteration','state', 'action', 'pred'))

            for i, sa_pair in enumerate(all_test_state_action_pairs):
                state, action = sa_pair[0], sa_pair[1]
                f1_calculation_data.append((iterations, 'our_approach', state, action, state_pred[i]))

                f.write("{},{},{},{}\n".format(iterations, state, action, state_pred[i]))
        f.close()

        test_acc, test_fpr, test_fnr = get_accuracy(test_grid, is_oracle=False)
        num_severe_nse, num_mild_nse, trajectory_lengths, test_visitation, mean_reward, reward_std = get_nse_encountered(test_grid, test_grid_agent_policy)
        f1_score, accuracy = get_f1_and_accuracy(test_grid, f1_calculation_data)
        with open(output_dir+'main_approach_test_grid.csv', 'a') as f:
            if os.stat(output_dir+'main_approach_test_grid.csv').st_size == 0:
                f.write("{},{},{},{},{},{},{},{},{}\n".format(*to_write))
            f.write("{},{},{},{},{},{},{},{},{}\n".format(iterations, test_map_name, test_acc, test_fpr, test_fnr, np.mean(num_severe_nse), np.std(num_severe_nse), np.mean(num_mild_nse), np.std(num_mild_nse)))
            f.close()

        with open(output_dir+'nse_count.csv', 'a') as f:
            if os.stat(output_dir+'nse_count.csv').st_size == 0:
                f.write("{},{},{},{},{}\n".format('env', 'iteration', '#severe_nse', '#mild_nse', 'traj_len'))
            for idx in range(len(trajectory_lengths)):
                f.write("{},{},{},{},{}\n".format(test_map_name, iterations, num_severe_nse[idx], num_mild_nse[idx], trajectory_lengths[idx]))
            f.close()

        with open(output_dir+'/f1_and_accuracy.csv', 'a') as f1:
            if os.stat(output_dir+'/f1_and_accuracy.csv').st_size == 0:
                f1.write("{},{},{},{},{}\n".format('iteration', 'method', 'map_name', 'f1_score', 'accuracy'))
            f1.write("{},{},{},{},{}\n".format(iterations, 'our_approach', test_map_name, f1_score, accuracy))
            f1.close()

        with open(output_dir+'/all_methods_rewards.csv', 'a') as f:
            if os.stat(output_dir+'/all_methods_rewards.csv').st_size == 0:
                f.write('iterations, test_map_name, mean_reward, reward_std\n')
            f.write('{}, {}, {}, {}\n'.format(iterations, test_map_name, mean_reward, reward_std))
            f.close()
