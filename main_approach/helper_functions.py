import numpy as np, random, math, itertools
from scipy.stats import entropy
from environments.lrtdp import lrtdp
from main_approach.approval import *
from main_approach.corrections import *
from main_approach.dam import *
from main_approach.ranking import *
from main_approach.reward_pred_model import predict
from environments.nse_analysis import *
from environments.helper_functions import *
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

def get_agent_initial_sa_labels(grid):
    m = []
    all_states = grid.all_states
    for state in all_states:
        state_m = []
        all_actions = range(grid.num_actions)
        if grid.domain=='bp':
            all_actions = grid.get_actions(state)
        for _ in all_actions:
            state_m.append(0)
        m.append(state_m)
    return m

def get_cluster_info_gain(clusters, all_states, p_cap, q_cap, m, prev_cs_id):
    prev_critical_states = {}
    for id in prev_cs_id:
        state = tuple(all_states[id])
        prev_critical_states[state] = [p_cap[id], q_cap[id], m[id]]
    cluster_p_q = {}
    for item in clusters:
        cluster_p_q[item] = []
        cluster_p_cap, cluster_q_cap, cluster_m_cap = [], [], []
        for i in clusters[item]:
            i = tuple(i)
            if i in prev_critical_states.keys():
                cluster_p_cap.append(prev_critical_states[i][0])
                cluster_q_cap.append(prev_critical_states[i][1])
                cluster_m_cap.append(prev_critical_states[i][2])
        cluster_p_q[item].append([cluster_p_cap, cluster_q_cap, cluster_m_cap])

    cluster_info_gain = []
    for item in cluster_p_q:
        cluster_p = cluster_p_q[item][0][0]
        cluster_q = cluster_p_q[item][0][1]
        cluster_m = cluster_p_q[item][0][2]

        all_labels = [0, 1, 2]
        cluster_p_dist = get_dist(cluster_p, all_labels)
        cluster_m_dist = get_dist(cluster_m, all_labels)

        gain_val = 0.001
        eps = 1e-10
        for j in range(len(cluster_p_dist)):
            pk, qk = cluster_p_dist[j], cluster_m_dist[j]
            val_k = entropy(pk+eps, qk+eps)
            gain_val += val_k
        gain_val = gain_val/len(cluster_p_dist)
        cluster_info_gain.append(gain_val)
    return cluster_info_gain

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def jaccard_distance(a, b):
    intersection = sum(x == y for x, y in zip(a, b))
    union = len(a) + len(b) - intersection
    return 1 - intersection / union

def assign_points_to_centroids(data, centroids):
    clusters = {i: [] for i in range(len(centroids))}
    labels = []
    for point in data:
        distances = [jaccard_distance(point, centroid) for centroid in centroids]
        min_distance_index = distances.index(min(distances))
        clusters[min_distance_index].append(point)
        labels.append(min_distance_index)
    return clusters, labels

def recompute_centroids(clusters):
    centroids = []
    for points in clusters.values():
        centroid = tuple(sum(col) / len(col) for col in zip(*points)) if points else None
        centroids.append(centroid)
    return centroids

def initialize_centroids_kmeans_plusplus(data, k):
    random.seed(42)
    centroids = [random.choice(data)]
    for _ in range(1, k):
        distances = [min(euclidean_distance(point, c) for c in centroids) for point in data]
        total = sum(distances)
        probabilities = [d / total for d in distances]
        cumulative_probabilities = list(itertools.accumulate(probabilities))
        r = random.random()
        for i, cp in enumerate(cumulative_probabilities):
            if r < cp:
                centroids.append(data[i])
                break
    return centroids

def handle_empty_clusters(clusters, centroids, data):
    random.seed(42)
    for i, centroid in enumerate(centroids):
        if centroid is None:
            centroids[i] = random.choice(data)
            clusters[i].append(centroids[i])
    return clusters, centroids

def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids_kmeans_plusplus(data, k)
    for _ in range(max_iterations):
        clusters, labels = assign_points_to_centroids(data, centroids)
        clusters, centroids = handle_empty_clusters(clusters, centroids, data)
        new_centroids = recompute_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters, centroids, labels

def k_centers(X, k):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    n_samples = X.shape[0]
    centers = [np.random.choice(n_samples)]
    for _ in range(1, k):
        distances = pairwise_distances(X, X[centers]).min(axis=1)
        new_center = np.argmax(distances)
        centers.append(new_center)
    distances = pairwise_distances(X, X[centers])
    labels = np.argmin(distances, axis=1)
    return centers, labels

def cluster_states(grid, k, cluster_algo='kmeans'):
    all_states = grid.all_states
    if grid.domain=='bp':
        formatted_data = [(int(bool1), int(bool2), int(bool3)) for coord, bool1, bool2, bool3 in all_states] # Format all_states to all numeric
    else:
        formatted_data = [(int(bool1), int(bool2)) for coord, bool1, bool2 in all_states] # Format all_states to all numeric
    if cluster_algo=='kcenters':
        formatted_data_list = [list(tup) for tup in formatted_data]
        _, labels = k_centers(formatted_data_list, k)
    elif cluster_algo=='dbscan':
        formatted_data_list = [list(tup) for tup in formatted_data]
        db = DBSCAN(eps=0.3, min_samples=10).fit(formatted_data_list)
        labels = db.labels_
    elif cluster_algo=='kmeans':
        _, _, labels = k_means(formatted_data, k)
    return labels

def sample_critical_states(grid, t, p_cap, q_cap, m, prev_critical_states, labels, n_clusters):
    all_states = grid.all_states
    k = n_clusters # Number of clusters
    n = int(0.05*len(all_states)) # Total number of states to sample
    clusters = {}
    i = 0
    for item in labels:
        if item in clusters:
                clusters[item].append(all_states[i])
        else:
                clusters[item] = [all_states[i]]
        i +=1

    cluster_sampling_wts = []
    if p_cap==[] and q_cap==[]:
        for _ in clusters:
            wt = int(n/k)
            cluster_sampling_wts.append(wt)
            samples_from_each_cluster = cluster_sampling_wts
        i = 0
        while sum(samples_from_each_cluster) != n:
            if sum(samples_from_each_cluster) > n:
                samples_from_each_cluster[i] -= 1
            elif sum(samples_from_each_cluster) < n:
                samples_from_each_cluster[i] += 1
            i += 1
    elif p_cap!=[] and q_cap!=[]:
        clusters_info_gain = get_cluster_info_gain(clusters, all_states, p_cap, q_cap, m, prev_critical_states)
        clusters_info_gain = np.array(clusters_info_gain)
        weights = clusters_info_gain / clusters_info_gain.sum()
        cluster_sampling_wts = weights.tolist()
        samples_from_each_cluster = [1] * n_clusters
        remaining_samples = n - sum(samples_from_each_cluster)
        for item in clusters:
            n_sample_from_each_cluster = math.ceil(cluster_sampling_wts[item]*remaining_samples)
            samples_from_each_cluster[item] += n_sample_from_each_cluster
        sorted_weights = np.argsort(weights)
        i = 0
        while sum(samples_from_each_cluster) != n:
            if sum(samples_from_each_cluster) > n:
                samples_from_each_cluster[sorted_weights[i]] -= 1
            elif sum(samples_from_each_cluster) < n:
                samples_from_each_cluster[sorted_weights[-i]] += 1
            i += 1
    sampled_state_indices = []
    for item in clusters:
        sampled_points = np.random.choice(len(clusters[item]), size=samples_from_each_cluster[item], replace=False)
        sampled_states = []
        for pt in sampled_points:
            state = clusters[item][pt]
            sampled_states.append(state)
        for state in sampled_states:
            if samples_from_each_cluster[item]>0:
                point_idx = all_states.index(state)
                sampled_state_indices.append(point_idx)
                samples_from_each_cluster[item] -= 1
            elif samples_from_each_cluster[item]==0:
                break
    return sampled_state_indices

def sample_random_critical_states(grid, is_random=False):
    all_states = grid.getStateFactorRep()
    if grid.domain=='vase':
        if is_random:
            num_to_sample = int((0.05) * len(all_states))
            state_indices = [idx for idx, state in enumerate(all_states)]
            sampled_indices = random.sample(state_indices, num_to_sample)
        else:
            num_to_sample = int((0.05/3) * len(all_states))
            floor_indices = [idx for idx, state in enumerate(all_states) if (state[1] == True and state[2] == False)]
            carpet_indices = [idx for idx, state in enumerate(all_states) if (state[1] == True and state[2] == True)]
            free_indices = [idx for idx, state in enumerate(all_states) if (state[1] == False and state[2] == False)]
            sampled_floor_indices = random.sample(floor_indices, num_to_sample)
            sampled_carpet_indices = random.sample(carpet_indices, num_to_sample)
            sampled_free_indices = random.sample(free_indices, num_to_sample)
            sampled_indices = sampled_floor_indices + sampled_carpet_indices + sampled_free_indices
    elif grid.domain=='outdoor':
        if is_random:
            num_to_sample = int((0.05) * len(all_states))
            state_indices = [idx for idx, state in enumerate(all_states)]
            sampled_indices = random.sample(state_indices, num_to_sample)
        else:
            num_to_sample = int((0.05/3) * len(all_states))
            floor_indices = [idx for idx, state in enumerate(all_states) if (state[1] == True and state[2] == False)]
            puddle_indices = [idx for idx, state in enumerate(all_states) if (state[2] == True)]
            free_indices = [idx for idx, state in enumerate(all_states) if (state[1] == False and state[2] == False)]
            sampled_floor_indices = random.sample(floor_indices, num_to_sample)
            sampled_puddle_indices = random.sample(puddle_indices, num_to_sample)
            sampled_free_indices = random.sample(free_indices, num_to_sample)
            sampled_indices = sampled_floor_indices + sampled_puddle_indices + sampled_free_indices
    else:
        if is_random:
            num_to_sample = int((0.05) * len(all_states))
            state_indices = [idx for idx, state in enumerate(all_states)]
            sampled_indices = random.sample(state_indices, num_to_sample)
        else:
            num_to_sample = int((0.05/4) * len(all_states))
            set1 = [idx for idx, state in enumerate(all_states) if (state[1] == True and state[2] == False and state[3] == True)]
            set2 = [idx for idx, state in enumerate(all_states) if (state[1] == True and state[2] == False and state[3] == False)]
            set3 = [idx for idx, state in enumerate(all_states) if (state[1] == True and state[2] == True)]
            set4 = [idx for idx, state in enumerate(all_states) if (state[1] == False)]
            sampled_set1_indices = random.sample(set1, num_to_sample)
            sampled_set2_indices = random.sample(set2, num_to_sample)
            sampled_set3_indices = random.sample(set3, num_to_sample)
            sampled_set4_indices = random.sample(set4, num_to_sample)
            sampled_indices = sampled_set1_indices + sampled_set2_indices + sampled_set3_indices + sampled_set4_indices
    return sampled_indices

def get_feedback_probs():
    # feedbacks: corr, app, dam, ann_corr, ann_app
    probs = [0.7, 0.5, 0.6, 0.65, 0.5, 0.55]
    return probs

def get_feedback_costs():
    costs = [7, 8, 5, 6, 7, 5]
    return costs

def get_chosen_feedback(feedback_ch, baseline=None):
    probs = get_feedback_probs()
    if baseline=='costs-sensitive':
        return feedback_ch
    fb_prob = probs[feedback_ch]
    choice = np.random.choice([None, feedback_ch], 1, p=[(1-fb_prob), fb_prob])
    return choice

def update_agent_policy(grid, model):
    state_preds = predict(grid, model)
    grid_agent_policy = lrtdp(grid, is_oracle=False, use_cache=True)
    return state_preds, grid_agent_policy

def collect_oracle_feedback(grid, critical_states, feedback_ch, oracle_policy, initial_agent_policy):
    if feedback_ch==0:
        get_corr_feedback(grid, critical_states, oracle_policy, initial_agent_policy)
    elif feedback_ch==1:
        get_app_feedback(grid, critical_states, oracle_policy, initial_agent_policy)
    elif feedback_ch==2:
        get_dam_feedback(grid, critical_states, oracle_policy, initial_agent_policy)
    elif feedback_ch==3:
        get_ann_corr_feedback(grid, critical_states, oracle_policy, initial_agent_policy)
    elif feedback_ch==4:
        get_ann_app_feedback(grid, critical_states, oracle_policy, initial_agent_policy)
    elif feedback_ch==5:
        get_rank_feedback(grid, critical_states, oracle_policy, initial_agent_policy)

def approximate_p_cap(grid):
    all_states = grid.getStateFactorRep()
    p_cap = []
    for state in all_states:
        state_vals = []
        state = tuple(state)
        all_actions = range(grid.num_actions)
        if grid.domain=='bp':
            all_actions = grid.get_actions(state)
        for action in all_actions:
            action = int(action)
            if (state, action) not in grid.learned_reward_cache:
                grid.learned_reward_cache[(state, action)] = 0
            state_vals.append(grid.learned_reward_cache[(state, action)])
        if state_vals!=[]:
            p_cap.append(state_vals)
    return p_cap

def get_dist(dist, all_labels):
    new_dist = []
    for state_vals in dist:
        unique_labels, label_counts = np.unique(state_vals, return_counts=True)
        label_prob = np.zeros(len(all_labels))
        label_prob[unique_labels] = label_counts/len(state_vals)
        new_dist.append(label_prob)
    return new_dist

def update_feedback_gain(p_cap, q_cap, feedback_ch, f_G, critical_states):
    cs_p_cap, cs_q_cap = [], []
    for id in critical_states:
        cs_p_cap.append(p_cap[id])
        cs_q_cap.append(q_cap[id])
    all_labels = [0, 1, 2]
    p_cap_dist = get_dist(cs_p_cap, all_labels)
    q_cap_dist = get_dist(cs_q_cap, all_labels)
    gain_val = 0.001
    eps = 1e-10
    for j in range(len(p_cap_dist)):
        pk, qk = p_cap_dist[j], q_cap_dist[j]
        val_k = entropy(pk+eps, qk+eps)
        gain_val += val_k
    gain_val = gain_val/len(p_cap_dist)
    f_G[feedback_ch] = gain_val
    return f_G

def write_val_to_file(all_fg, all_budget, output_dir, tot_budget):
    filename = Path(output_dir+str(tot_budget)+'.csv')
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'a') as f1:
        if os.stat(filename).st_size == 0:
            f1.write("budget, fg\n")
        for i in range(len(all_budget)):
            f1.write('{},{}\n'.format(all_budget[i], all_fg[i]))
        f1.close()

def write_config(filename, f_costs, f_probs):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'a') as f1:
        if os.stat(filename).st_size == 0:
            f1.write("variables, value\n")
        f1.write('feedback cost (corr, app, dam, ann-corr, ann-app), {}\n'.format(f_costs))
        f1.write('feedback probs (corr, app, dam, ann-corr, ann-app), {}\n'.format(f_probs))
        f1.close()
