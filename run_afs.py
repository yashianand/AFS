from environments.helper_functions import *
from environments.lrtdp import lrtdp
from main_approach.algo import learn
from ai_safety_gridworlds.environments import side_effects_vase
from environments.oracle import *
import sys, time
from main_approach.helper_functions import *
from main_approach.test_grids_prediction import pred_test_grid

output_fn = '/afs/'

test_map_names = []
if sys.argv[1]=='bp':
    filename = 'fmdp/boxpushing/gameArt_size15.txt'
    iterations = [400, 800, 1200, 1600, 2000, 2500]
    train_map_name = 0
    output_dir = "outputs"+output_fn
    is_safety_grid = False
elif sys.argv[1]=='ua-vase':
    filename = 'fmdp/unavoidableNSE_vase/gameArt_size15.txt'
    iterations = [400, 800, 1200, 1600, 2000, 2500]
    train_map_name = side_effects_vase
    output_dir = "outputs"+output_fn
    is_safety_grid = True
elif sys.argv[1]=='outdoor':
    filename = 'fmdp/outdoor_robot/gameArt_size15.txt'
    iterations = [400, 800, 1200, 1600, 2000, 2500]
    train_map_name = 0
    output_dir = "outputs"+output_fn
    is_safety_grid = 'outdoor'
else:
    filename = 'fmdp/gridworlds/gameArt_size15.txt'
    iterations = [400, 800, 1200, 1600, 2000, 2500]
    train_map_name = side_effects_vase
    output_dir = "outputs"+output_fn
    is_safety_grid = True

if is_safety_grid==True:
    side_effects_vase.GAME_ART = eval(open(filename).read())
    t1 = time.time()
    train_grid, train_grid_oracle_policy = get_oracle_performance(train_map_name, iterations, output_dir, filename, is_train=True, level=0, is_safety_grid=is_safety_grid, is_lrtdp=True)
    initial_agent_policy = lrtdp(train_grid, is_oracle=False)
    for level in range(1, 4):
        test_grid, test_grid_oracle_policy = get_oracle_performance(train_map_name, iterations, output_dir, filename, is_train=False, level=level, is_safety_grid=is_safety_grid, is_lrtdp=True)
        test_map_names.append(level)
else:
    train_grid, train_grid_oracle_policy = get_oracle_performance(train_map_name, iterations, output_dir, filename, is_train=True, level=None, is_safety_grid=is_safety_grid)
    initial_agent_policy = lrtdp(train_grid, is_oracle=False)
    num_severe_nse, num_mild_nse, trajectory_lengths, oracle_visitation, mean_reward, reward_std = get_nse_encountered(train_grid, initial_agent_policy)
    for test_map_name in range(1, 4):
        test_grid, test_grid_oracle_policy = get_oracle_performance(test_map_name, iterations, output_dir, filename, is_train=False, level=None, is_safety_grid=is_safety_grid)
        test_map_names.append(test_map_name)
n_clusters = 3
labels = cluster_states(train_grid, n_clusters, cluster_algo='kcenters')
for i in iterations:
    model = learn(train_grid, train_grid_oracle_policy, initial_agent_policy, labels, output_dir, n_clusters, tot_budget=i, baseline=None)
    pred_test_grid(test_map_names, model, output_dir, filename, iterations=i, is_safety_grid=is_safety_grid)
write_config(output_dir+"config.csv", get_feedback_costs(), get_feedback_probs())
