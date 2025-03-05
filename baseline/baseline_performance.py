import os
from environments.lrtdp import lrtdp
from baseline.helper_functions import *
from environments.nse_analysis import *

def get_performance(best_theta, num_feedback, output_dir, filename, is_safety_grid):

    for test_map_name in range(1, 4):
        test_grid = readMDPfile(filename, test_map_name, is_safety_grid=is_safety_grid, is_oracle=False, is_lrtdp=True)
        test_grid.reset()
        test_grid.get_all_reward = True
        test_grid.is_baseline = True
        test_grid.calibration_reward = best_theta
        test_grid.get_all_reward = True
        test_grid_agent_policy = lrtdp(test_grid, is_oracle=False)
        num_severe_nse, num_mild_nse, _, test_visitation, mean_reward, reward_std = get_nse_encountered(test_grid, test_grid_agent_policy)
        to_write = ['map_name', 'num_feedback', 'selected_feedback', 'theta_est', 'avg_severe_nse', 'severe_nse_sd', 'avg_mild_nse', 'mild_nse_sd']
        with open(output_dir+'main_approach_test_grid.csv', 'a') as f:
            if os.stat(output_dir+'main_approach_test_grid.csv').st_size == 0:
                f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(*to_write))
            f.write("{}, {}, {}, {}, {}, {}, {}\n".format(test_map_name, num_feedback, best_theta, np.mean(num_severe_nse), np.std(num_severe_nse), np.mean(num_mild_nse), np.std(num_mild_nse)))
            f.close()
        with open(output_dir+'/all_methods_rewards.csv', 'a') as f1:
            if os.stat(output_dir+'/all_methods_rewards.csv').st_size == 0:
                f1.write('map_name, num_feedback, selected_feedback, theta_est, mean_reward, reward_std\n')
            f1.write('{}, {}, {}, {}, {}, \n'.format(test_map_name, num_feedback, best_theta, mean_reward, reward_std))
            f1.close()
