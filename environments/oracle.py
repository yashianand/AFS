import numpy as np
import os
from environments.nse_analysis import *
from environments.helper_functions import *
from environments.VI_fmdp import valueIteration
from environments.lrtdp import lrtdp

def get_oracle_performance(env, iterations, output_dir, filename,is_train, level=None, is_safety_grid=False, is_lrtdp=True):
    map_name = level if is_safety_grid==True else env
    grid = readMDPfile(filename, map_name, is_safety_grid=is_safety_grid, is_oracle=True, is_lrtdp=is_lrtdp)
    print('map {} read complete'.format(map_name))
    to_write = ['env', 'iteration', 'method', 'accuracy', 'fpRate', 'fnRate', 'avg_severe_nse', 'severe_nse_sd', 'avg_mild_nse', 'mild_nse_sd']
    oracle_avg_severe_nse, oracle_avg_mild_nse = [], []
    oracle_severe_nse_sd,  oracle_mild_nse_sd = [], []
    oracle_accuracy, fpRate, fnRate = [], [], []
    if is_lrtdp:
        print('Running LRTDP for oracle...')
        oracle_policy = lrtdp(grid)
        print('is oracle: ', grid.is_oracle)
        print('Complete')
    else:
        print('Running VI for oracle...')
        _, oracle_policy = valueIteration(grid, is_oracle=True)
        print('Complete')

    acc, fpr, fnr = get_accuracy(grid, is_oracle=True)
    num_severe_nse, num_mild_nse, trajectory_lengths, oracle_visitation, mean_reward, reward_std = get_nse_encountered(grid, oracle_policy)

    for _ in iterations:
        oracle_accuracy.append(acc)
        fpRate.append(fpr)
        fnRate.append(fnr)
        oracle_avg_severe_nse.append(np.mean(num_severe_nse))
        oracle_severe_nse_sd.append(np.std(num_severe_nse))
        oracle_avg_mild_nse.append(np.mean(num_mild_nse))
        oracle_mild_nse_sd.append(np.std(num_mild_nse))


    if is_train==True:
        filename = output_dir+'/train_grid_values.csv'

    else:
        filename = output_dir+'/test_grid_values.csv'

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'a') as f:
        if os.stat(filename).st_size == 0:
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(*to_write))
        for i in range(len(iterations)):
            if is_safety_grid==True:
                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(level, iterations[i], 'oracle', oracle_accuracy[i], fpRate[i], fnRate[i], oracle_avg_severe_nse[i], oracle_severe_nse_sd[i], oracle_avg_mild_nse[i], oracle_mild_nse_sd[i]))
            else:
                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(env, iterations[i], 'oracle', oracle_accuracy[i], fpRate[i], fnRate[i], oracle_avg_severe_nse[i], oracle_severe_nse_sd[i], oracle_avg_mild_nse[i], oracle_mild_nse_sd[i]))
    f.close()

    filename = Path(output_dir+'/oracle_rewards.csv')
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'a') as f:
        if os.stat(filename).st_size == 0:
            f.write('iterations, test_map_name, mean_reward, reward_std\n')
        f.write('{}, {}, {}, {}\n'.format('oracle', level, mean_reward, reward_std))

    filename = Path(output_dir+'/oracle_nse_count.csv')
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'a') as f:
        if os.stat(filename).st_size == 0:
            f.write('env, #severe_nse, #mild_nse, traj_len\n')
        for idx in range(len(trajectory_lengths)):
            f.write('{}, {}, {}, {}\n'.format(map_name, num_severe_nse[idx], num_mild_nse[idx], trajectory_lengths[idx]))

    return grid, oracle_policy
