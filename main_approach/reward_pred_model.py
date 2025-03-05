from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn import metrics
import numpy as np
from copy import deepcopy

def update_reward_model(train_grid):
    seen_state_features = []
    seen_state_loc = []
    seen_penalty = []
    seen_state_action = []
    for keys, value in train_grid.learned_reward_cache.items():
        seen_state_loc.append(keys[0][0])
        seen_state_action.append(keys[1])
        if train_grid.domain=='vase' or train_grid.domain=='outdoor':
            seen_state_features.append([0 if keys[0][1]==False else 1, 0 if keys[0][2]==False else 1, keys[1]])
        else:
            seen_state_features.append([0 if keys[0][1]==False else 1, 0 if keys[0][2]==False else 1, 0 if keys[0][3]==False else 1, keys[1]])
        seen_penalty.append(value)
    final_model = learn_reward_model(seen_state_features, seen_penalty, model_type='classification')
    return final_model

def train_folds(model,x_train_, y_train_, x_test_, y_test_):
    model.fit(x_train_,y_train_)
    test_label = model.predict(x_test_)
    mse = metrics.mean_squared_error(test_label, y_test_)
    return mse

def learn_reward_model(X, y, model_type='regression'):
    zero_count = y.count(0)
    one_count = y.count(1)
    two_count = y.count(2)
    X = np.array(X)
    y = np.array(y)
    params = {
                "max_depth": [None, 2, 5, 10],
                "max_features":['sqrt', 'log2', None],
                "min_samples_leaf": [1, 5, 10, 50],
                "max_leaf_nodes": [None, 5, 10, 20],
                "n_estimators": [50, 100, 200, 500, 900],
                "class_weight": ["balanced", None],
            }

    num_splits = 2
    if zero_count < 2 or one_count < 2 or two_count < 2:
        num_splits = 1
    best_score = -1
    best_params = {}
    candidate_params = list(ParameterSampler(param_distributions=params, n_iter=10))
    init_model = RandomForestRegressor()
    if model_type == 'classification':
        init_model = RandomForestClassifier()
    model = deepcopy(init_model)
    for parameters in candidate_params:
        model.set_params(**parameters)
        if(num_splits > 1):
            cv =  StratifiedKFold(n_splits=num_splits, shuffle=True)
            cv_scores  = []
            for train, test in cv.split(X, y[:]):
                x_train_, x_test_, y_train_, y_test_ =  X[train], X[test], y[train], y[test]
                score = train_folds(model, x_train_, y_train_, x_test_, y_test_)
                cv_scores.append(score)
            avg_score = float(sum(cv_scores))/len(cv_scores)
            if(avg_score > best_score):
                best_params = parameters
                best_score = avg_score
        else:
            best_params = parameters
            break
    final_model = deepcopy(model)
    final_model.set_params(**best_params)
    final_model.fit(X, y)
    return final_model

def get_state_action_successors(grid):
    state_action_pairs = []
    for state in grid.getStateFactorRep():
        all_actions = range(grid.num_actions)
        if grid.domain=='bp':
            all_actions = grid.get_actions(state)
        for action in all_actions:
            if grid.get_successors(state, action)!=[]:
                state_action_pairs.append((tuple(state), action))
    return state_action_pairs

def predict(grid, model):
    all_test_state_action_pairs = get_state_action_successors(grid)
    unseen_state_feature = []
    unseen_state_loc = []
    unseen_state_action = []
    for sa_pair in all_test_state_action_pairs:
        unseen_state_loc.append((sa_pair[0][0]))
        if grid.domain=='vase' or grid.domain=='outdoor':
            unseen_state_feature.append([0 if sa_pair[0][1]==False else 1, 0 if sa_pair[0][2]==False else 1, sa_pair[1]])
        else:
            unseen_state_feature.append([0 if sa_pair[0][1]==False else 1, 0 if sa_pair[0][2]==False else 1, 0 if sa_pair[0][3]==False else 1, sa_pair[1]])
        unseen_state_action.append(sa_pair[1])
    state_pred = model.predict(np.array(unseen_state_feature))
    for i in range(len(unseen_state_loc)):
        if grid.domain=='vase' or grid.domain=='outdoor':
            state = ((unseen_state_loc[i]), True if unseen_state_feature[i][0]==1 else False, True if unseen_state_feature[i][1]==1 else False)
        else:
            state = ((unseen_state_loc[i]), True if unseen_state_feature[i][0]==1 else False, True if unseen_state_feature[i][1]==1 else False, True if unseen_state_feature[i][2]==1 else False)
        action = unseen_state_action[i]
        if state_pred[i] == 0:
            grid.agent_reward_cache[(state, action)] = grid.get_reward(state, action, is_oracle=False, is_lrtdp=True)
        elif state_pred[i] == 1:
            grid.agent_reward_cache[(state, action)] = grid.get_reward(state, action, is_oracle=False, is_lrtdp=True) + 5
        elif state_pred[i] == 2:
            grid.agent_reward_cache[(state, action)] = grid.get_reward(state, action, is_oracle=False, is_lrtdp=True) + 10
    return state_pred

def pred_cs_and_get_qcap(grid, cs_ids, model):
    all_states = grid.getStateFactorRep()
    unseen_samples = []
    sa_pair = []
    q_cap = []
    for s_id in cs_ids:
        s = tuple(all_states[s_id])
        all_actions = range(grid.num_actions)
        if grid.domain=='bp':
            all_actions = grid.get_actions(s)
        for a in all_actions:
            if grid.domain=='vase' or grid.domain=='outdoor':
                unseen_samples.append([0 if s[1]==False else 1, 0 if s[2]==False else 1, a])
            else:
                unseen_samples.append([0 if s[1]==False else 1, 0 if s[2]==False else 1, 0 if s[3]==False else 1, a])
            sa_pair.append((s, a))
    state_pred = model.predict(np.array(unseen_samples))
    for i, sa in enumerate(sa_pair):
        grid.critical_state_preds[sa] = state_pred[i]
    for state in all_states:
        state = tuple(state)
        state_vals = []
        all_actions = range(grid.num_actions)
        if grid.domain=='bp':
            all_actions = grid.get_actions(state)
        for a in all_actions:
            if (state, a) in grid.critical_state_preds:
                state_vals.append(grid.critical_state_preds[(state, a)])
            else:
                state_vals.append(grid.learned_reward_cache[(state, a)])
        if state_vals!=[]:
            q_cap.append(state_vals)
    return q_cap
