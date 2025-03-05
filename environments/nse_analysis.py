from environments.helper_functions import simulate_trajectory
from environments.get_grid import *

def get_nse_states(grid):
    factored_states = grid.getStateFactorRep()
    nse_states = []
    for state in factored_states:
        state = tuple(state)
        if state[1]==True and state[2]==False:
            nse_states.append(state)
    return nse_states

def get_accuracy(grid, is_oracle=False):
    factored_states = grid.getStateFactorRep()
    fp, fn =0, 0
    tp, tn = 0, 0
    accuracy = 0
    for state in factored_states:
        if grid.domain=='bp':
            all_actions = grid.get_actions(state)
        else:
            all_actions = range(grid.num_actions)
        for action in all_actions:
            if is_oracle:
                reward = grid.get_reward(state, action, is_oracle=True)
            else:
                reward = grid.agent_reward_cache[(state, action)]
            if reward > 1 and ((state[1]==True and state[2]==True) or (state[1]==False and state[2]==False)):
                fp += 1
            elif reward <= 1 and  (state[1]==True and state[2]==False):
                fn += 1
            elif reward <= 1 and ((state[1]==True and state[2]==True) or (state[1]==False and state[2]==False)):
                tn += 1
            elif reward > 1 and (state[1]==True and state[2]==False):
                tp += 1
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)
    return accuracy, fpr, fnr

def  get_nse_encountered(grid, policy, num_trajectories=100):
    times_visited = np.zeros((grid.rows, grid.cols))
    severe_nse_per_trajectory, mild_nse_per_trajectory = [], []
    rewards = []
    trajectory_lengths = []
    for _ in range(num_trajectories):
        num_severe_nse = 0
        num_mild_nse = 0
        trajectory = []
        traj_rewards, trajectory = simulate_trajectory(grid, policy)
        trajectory_lengths.append(len(trajectory))
        rewards.append(traj_rewards)
        for step in range(len(trajectory)):
            state = trajectory[step][0]
            times_visited[state[0][0]][state[0][1]] += 1
            if grid.domain=='vase':
                if state[1]==True and state[2]==False:
                    num_severe_nse += 1
                elif state[1]==True and state[2]==True:
                    num_mild_nse += 1
            elif grid.domain=='outdoor':
                if state[1]==False and state[2]==False:
                    num_mild_nse += 1
                elif state[1]==False and state[2]==True:
                    num_severe_nse += 1
            elif grid.domain=='bp':
                if state[0]!=grid.box_loc:
                    if state[1]==True and state[2]==False and state[3]==True:
                        num_severe_nse += 1
                    elif state[1]==True and state[2]==False and state[3]==False:
                        num_mild_nse += 1
        severe_nse_per_trajectory.append(num_severe_nse)
        mild_nse_per_trajectory.append(num_mild_nse)
    return severe_nse_per_trajectory, mild_nse_per_trajectory, trajectory_lengths, times_visited, np.mean(rewards), np.std(rewards)

def analyze_visited_states(grid, times_visited):
    states = grid.getStateFactorRep()
    for state in states:
        state = tuple(state)
        for key, value in grid.learned_reward_cache:
            if key == state:
                times_visited[state[0]] += 1
    return times_visited

def get_agent_visitation(grid, agent_demos, output_dir, method, budget):
    times_visited = np.zeros((grid.rows, grid.cols))
    for state, action in agent_demos.items():
        times_visited[state[0][0]][state[0][1]] += 1
    plt.clf()
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title('State Frequency Heatmap')
    ax = sns.heatmap(times_visited, linewidths=1, linecolor='black', square=True, annot=True, fmt='.0f', cmap=['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494'])
    for row in range(grid.rows):
        for col in range(grid.cols):
            if grid.grid[row][col] == '$':
                ax.add_patch(patches.Rectangle((col, row), 1, 1, ec='red', fc='none', lw=1, hatch='//'))
            elif grid.grid[row][col] == 'V':
                ax.add_patch(patches.Rectangle((col, row), 1, 1, ec='orange', fc='none', lw=1, hatch='//'))
    plt.savefig(output_dir+method+'-agent_visition-'+str(budget)+'.png')
    plt.close()

def get_f1_and_accuracy(grid, prediction_entries):
    iteration = prediction_entries[0][0]
    method = prediction_entries[0][1]
    domain = grid.domain
    tp = {'no-nse': 0, 'mild-nse': 0, 'severe-nse': 0}
    fp = {'no-nse': 0, 'mild-nse': 0, 'severe-nse': 0}
    fn = {'no-nse': 0, 'mild-nse': 0, 'severe-nse': 0}
    for i in range(len(prediction_entries)):
        state = prediction_entries[i][2]
        action = prediction_entries[i][3]
        state_pred = prediction_entries[i][4]

        if domain=='vase' or domain=='ua-vase':
            # severe nse
            if state[1]==True and state[2]==False:
                if state_pred==2:
                    tp['severe-nse'] += 1
                elif state_pred==1:
                    fn['severe-nse'] += 1
                    fp['mild-nse'] += 1
                elif state_pred==0:
                    fn['severe-nse'] += 1
                    fp['no-nse'] += 1
            # mild nse
            elif state[1]==True and state[2]==True:
                if state_pred==1:
                    tp['mild-nse'] += 1
                elif state_pred==2:
                    fn['mild-nse'] += 1
                    fp['severe-nse'] += 1
                elif state_pred==0:
                    fn['mild-nse'] += 1
                    fp['no-nse'] += 1
            # no nse
            else:
                if state_pred==0:
                    tp['no-nse'] += 1
                elif state_pred==1:
                    fn['no-nse'] += 1
                    fp['mild-nse'] += 1
                elif state_pred==2:
                    fn['no-nse'] += 1
                    fp['severe-nse'] += 1
        elif domain=='outdoor':
            # severe nse
            if state[1]==False and state[2]==True:
                if state_pred==2:
                    tp['severe-nse'] += 1
                elif state_pred==1:
                    fn['severe-nse'] += 1
                    fp['mild-nse'] += 1
                elif state_pred==0:
                    fn['severe-nse'] += 1
                    fp['no-nse'] += 1
            # mild nse
            elif state[1]==False and state[2]==False:
                if state_pred==1:
                    tp['mild-nse'] += 1
                elif state_pred==2:
                    fn['mild-nse'] += 1
                    fp['severe-nse'] += 1
                elif state_pred==0:
                    fn['mild-nse'] += 1
                    fp['no-nse'] += 1
            # no nse
            else:
                if state_pred==0:
                    tp['no-nse'] += 1
                elif state_pred==1:
                    fn['no-nse'] += 1
                    fp['mild-nse'] += 1
                elif state_pred==2:
                    fn['no-nse'] += 1
                    fp['severe-nse'] += 1
        elif domain=='bp':
            # severe nse
            if state[1]==True and state[2]==False and state[3]==True:
                if state_pred==2:
                    tp['severe-nse'] += 1
                elif state_pred==1:
                    fn['severe-nse'] += 1
                    fp['mild-nse'] += 1
                elif state_pred==0:
                    fn['severe-nse'] += 1
                    fp['no-nse'] += 1
            # mild nse
            elif state[1]==True and state[2]==False and state[3]==False:
                if state_pred==1:
                    tp['mild-nse'] += 1
                elif state_pred==2:
                    fn['mild-nse'] += 1
                    fp['severe-nse'] += 1
                elif state_pred==0:
                    fn['mild-nse'] += 1
                    fp['no-nse'] += 1
            # no nse
            else:
                if state_pred==0:
                    tp['no-nse'] += 1
                elif state_pred==1:
                    fn['no-nse'] += 1
                    fp['mild-nse'] += 1
                elif state_pred==2:
                    fn['no-nse'] += 1
                    fp['severe-nse'] += 1
    precision = {'no-nse': 0, 'mild-nse': 0, 'severe-nse': 0}
    recall = {'no-nse': 0, 'mild-nse': 0, 'severe-nse': 0}
    f1_score = {'no-nse': 0, 'mild-nse': 0, 'severe-nse': 0}
    nse_severity = ['no-nse', 'mild-nse', 'severe-nse']
    for nse in nse_severity:
        try:
            precision[nse] = tp[nse] / (tp[nse]+fp[nse])
        except ZeroDivisionError:
            precision[nse] = 0
        try:
            recall[nse] = tp[nse] / (tp[nse]+fn[nse])
        except ZeroDivisionError:
            recall[nse] = 0
        try:
            f1_score[nse] = (2*precision[nse]*recall[nse]) / (precision[nse]+recall[nse])
        except ZeroDivisionError:
            f1_score[nse] = 0
    all_correct = sum(tp.values())
    tot_samples = len(prediction_entries)
    accuracy = all_correct / tot_samples
    return f1_score, accuracy
