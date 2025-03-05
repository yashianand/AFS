from ai_safety_gridworlds.environments import side_effects_vase
from environments.lrtdp import lrtdp
from environments.oracle import *
import sys
from baseline.beta_estimation import estimate_beta
from baseline.baseline_performance import get_performance
from baseline.helper_functions import get_random_theta
from baseline.al import active_feedback_selection

map_size = int(sys.argv[2])
output_fn = '/grid_size{}_baseline/'.format(str(map_size))
num_feedback = [400, 800, 1200, 1600, 2000, 2500]

if sys.argv[1] == 'bp':
    filename = 'fmdp/boxpushing/gameArt_size15.txt'
    train_map_name = 0
    output_dir = "new_ranking_method/boxpushing"+output_fn
    is_safety_grid = False
elif sys.argv[1]=='ua-vase':
    filename = 'fmdp/unavoidableNSE_vase/gameArt_size15.txt'
    train_map_name = side_effects_vase
    output_dir = "new_ranking_method/vase_unavoidable"+output_fn
    is_safety_grid = True
elif sys.argv[1]=='outdoor':
    filename = 'fmdp/outdoor_robot/gameArt_size15.txt'
    train_map_name = 0
    output_dir = "new_ranking_method/outdoor"+output_fn
    is_safety_grid = 'outdoor'
else:
    filename = 'fmdp/gridworlds/gameArt_size15.txt'
    train_map_name = side_effects_vase
    output_dir = "new_ranking_method/side_effects_vase"+output_fn
    is_safety_grid = True

if is_safety_grid==True:
    side_effects_vase.GAME_ART = eval(open(filename).read())
    train_grid, train_grid_oracle_policy = get_oracle_performance(train_map_name, [1], output_dir, filename, is_train=True, level=0, is_safety_grid=True, is_lrtdp=True)
    oracle_q_values = train_grid.oracle_q_values
    train_grid_agent_policy = lrtdp(train_grid, is_oracle=False)
    train_grid.oracle_q_values = oracle_q_values
else:
    train_grid, train_grid_oracle_policy = get_oracle_performance(train_map_name, [1], output_dir, filename, is_train=True, level=None, is_safety_grid=is_safety_grid, is_lrtdp=True)
    oracle_q_values = train_grid.oracle_q_values
    train_grid_agent_policy = lrtdp(train_grid, is_oracle=False)
    train_grid.oracle_q_values = oracle_q_values

train_grid.is_baseline = True
calibration_rewards = get_random_theta(train_grid, n=10)
unknown_reward = get_random_theta(train_grid, n=1)[0]

train_grid.calibration_reward = unknown_reward
oracle_policy = lrtdp(train_grid, is_oracle=True)
agent_policy = lrtdp(train_grid, is_oracle=False)

for n_feedback in num_feedback:
    beta_cap = estimate_beta(train_grid, n_feedback, calibration_rewards, output_dir)
    theta_val = active_feedback_selection(train_grid, beta_cap, oracle_policy, agent_policy, n_feedback, calibration_rewards, output_dir)
    get_performance(theta_val, n_feedback, output_dir, filename, is_safety_grid)
