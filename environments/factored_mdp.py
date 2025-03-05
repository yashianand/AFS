import numpy as np

class FactoredGridWorld:
    gridWorld = None
    def __init__(self, grid, is_oracle=False, is_lrtdp=True, calibration_reward=[]):
        self.grid = grid = np.asarray(grid, dtype='c')
        self.grid_list = [[c.decode('utf-8') for c in line] for line in self.grid.tolist()]
        self.actions = [0, 1, 2, 3, 4, 5] # left, up, right, down, pick, wrap&pick
        self.num_actions = len(self.actions)
        self.state = None
        self.terminal_state = None
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.num_states = self.rows * self.cols
        self.num_walls = sum(row.count('#') for row in self.grid_list)
        self.num_states_not_wall = self.num_states - self.num_walls
        self.agent_reward_cache = {}
        self.learned_reward_cache = {}
        self.oracle_demos = {}
        self.critical_state_preds = {}
        self.reward_cache = {}
        self.oracle_q_values = {}
        self.agent_q_values = {}
        self.is_oracle = is_oracle
        self.is_lrtdp = is_lrtdp
        self.calibration_reward = calibration_reward
        self.is_baseline = False
        self.get_all_reward = False
        self.start_state = (((np.where(self.grid == b'A')[0].item(), np.where(self.grid == b'A')[1].item()), False, False, False))
        self.box_loc = (np.where(self.grid == b'B')[0].item(), np.where(self.grid == b'B')[1].item())
        self.all_states = None
        self.domain = 'bp'
        self.reset()

    def reset(self, full_grid=True):
        if full_grid:
            self.agent_reward_cache = {}
            self.learned_reward_cache = {}
            self.oracle_demos = {}
            self.reward_cache = {}
            self.critical_state_preds = {}
            self.oracle_q_values = {}
            self.agent_q_values = {}
            all_states = self.getStateFactorRep()
            for state in all_states:
                all_actions = self.get_actions(state)
                for action in all_actions:
                    self.oracle_q_values[(state, action)] = 0.001
                    self.agent_q_values[(state, action)] = 0.001
        self.box_loc = (np.where(self.grid == b'B')[0].item(), np.where(self.grid == b'B')[1].item())
        self.terminal_state = (np.where(self.grid == b'G')[0].item(), np.where(self.grid == b'G')[1].item())
        self.state = (((np.where(self.grid == b'A')[0].item(), np.where(self.grid == b'A')[1].item()), False, False, False))
        self.all_states = self.getStateFactorRep()


    def getStateFactorRep(self):
        featureRep = []
        for i in range(self.rows):
            for j in range(self.cols):
                currState = self.grid_list[i][j]
                if currState != '#':
                    for b in [0, 1]:
                        if b==0:
                            w_vals = [0]
                        else:
                            w_vals = [0, 1]
                        for w in w_vals:
                            featureRep.append(((i, j), True if b==1 else False, True if w==1 else False, True if currState=='C' else False))
        return featureRep

    def step(self, action):
        terminal = False
        successors, succ_probabilities = self.get_successors(self.state, action)
        next_state_idx = np.random.choice(len(successors), p=succ_probabilities)
        self.state = successors[next_state_idx]
        if (self.state, action) in self.agent_reward_cache:
            reward = self.agent_reward_cache[(self.state, action)]
        else:
            reward = self.get_reward(self.state, action, self.is_oracle, self.is_lrtdp)
        if self.is_goal(self.state):
            terminal = True
        return successors[next_state_idx], reward, succ_probabilities[next_state_idx], terminal

    def getActionFactorRep(self, a):
        if a == 0: # left
            return (0,-1)
        elif a == 1: # up
            return (-1,0)
        elif a == 2: # right
            return (0,1)
        elif a == 3: # down
            return (1,0)

    def is_boundary(self, state):
        x, y = state
        return (x <= 0 or x >= self.rows-1 or y <= 0 or y >= self.cols-1 )

    def is_goal(self, state):
        if (state[0] == self.terminal_state) and (state[1] == True):
            return True
        return False

    def move(self, currFactoredState, action):
        state, loaded, wrapped, carpet = currFactoredState
        if action in [0, 1, 2, 3]:
            new_state = tuple(x + y for (x, y) in zip(state, self.getActionFactorRep(action)))
            if self.is_boundary(new_state):
                return currFactoredState, True
            else:
                if self.grid_list[new_state[0]][new_state[1]] == 'C':
                    return (new_state, loaded, wrapped, True), False
                else:
                    return (new_state, loaded, wrapped, False), False
        else:
            if action == 4:
                if state==self.box_loc and loaded==False:
                    return (state, True, wrapped, carpet), False
            elif action == 5:
                if loaded==True and wrapped==False:
                    return (state, True, True, carpet), False
            return currFactoredState, False


    def get_actions(self, state):
        if state[0] == self.box_loc:
            action_set = [0, 1, 2, 3, 4, 5]
        elif state[1] == True:
            action_set = [0, 1, 2, 3]
        else:
            action_set = [0, 1, 2, 3, 4, 5]
        return action_set

    def get_side_states(self, state, action):
        side_states =[]
        for a in [0, 1, 2, 3]:
            if a != action:
                new_state, is_wall = self.move(state, a)
                if not is_wall:
                    side_states.append(new_state)
                elif is_wall:
                    side_states.append(state)
        return side_states

    def get_transition(self, curr_state, action, next_state):
        succ_factored_state, is_wall = self.move(curr_state, action)
        if action in [0, 1, 2, 3]:
            sstates = self.get_side_states(curr_state, action)
            success_prob = 0.8
            fail_prob = 0.2/3

            if is_wall:
                transition_probs = []
                for feature_idx in range(len(curr_state)):
                    if (curr_state[feature_idx] == next_state[feature_idx]):
                        transition_probs.append(1)
                    else:
                        transition_probs.append(0)
                return np.prod(transition_probs)

            elif not is_wall:
                transition_probs = []
                if (next_state[0]==succ_factored_state[0]):
                    transition_probs.append(success_prob)
                    if (next_state[1]==succ_factored_state[1]):
                        transition_probs.append(1)
                    elif (next_state[1]!=succ_factored_state[1]):
                        transition_probs.append(0)
                    return np.prod(transition_probs)
                for side_state in sstates:
                    if (next_state[0]==side_state[0]):
                        state_count = sstates.count(next_state)
                        fail_prob *= state_count
                        transition_probs.append(fail_prob)
                        if (next_state[1]==side_state[1]):
                            transition_probs.append(1)
                        elif (next_state[1]!=side_state[1]):
                            transition_probs.append(0)
                        return np.prod(transition_probs)
        else:
            transition_probs = []
            if (next_state[0]==succ_factored_state[0]):
                transition_probs.append(1)
                if (next_state[1]==succ_factored_state[1]):
                    transition_probs.append(1)
                elif (next_state[1]!=succ_factored_state[1]):
                    transition_probs.append(0)
                if (next_state[2]==succ_factored_state[2]):
                    transition_probs.append(1)
                elif (next_state[2]!=succ_factored_state[2]):
                    transition_probs.append(0)
                return np.prod(transition_probs)

        return 0

    def get_possible_next_states(self, state):
        possible_states = set()
        action_set = self.get_actions(state)
        for action in action_set:
            next_state, _ = self.move(state, action)

            possible_states.add(next_state)
        return possible_states

    def get_successors(self, state, action):
        successors, succ_probabilities = [], []
        for next_state in self.get_possible_next_states(state):
            p = self.get_transition(state, action, next_state)
            if p > 0:
                successors.append(next_state)
                succ_probabilities.append(p)
        return successors, succ_probabilities

    def get_reward(self, state, action, is_oracle=False, is_lrtdp=None):
        state_reward = None
        is_lrtdp = self.is_lrtdp
        (x,y), loaded, wrapped, carpet = state
        if is_lrtdp==True:
            goal = 0
            severe_nse = 10
            mild_nse = 5
            if len(self.calibration_reward)!=0:
                nse=self.calibration_reward
            step_reward = 1
        if self.is_baseline==True:
            if self.is_goal(state) == True:
                return goal
            else:
                r = step_reward
                r += (nse[0]*loaded) + (nse[1]*wrapped) + (nse[2]*carpet)
                return r
        else:
            if action in [0,1,2,3,4,5]:
                if is_oracle or self.get_all_reward:
                    if self.is_goal(state) == True:
                        return goal
                    elif loaded==True and wrapped==False and carpet==True:
                        return severe_nse
                    elif loaded==True and wrapped==False and carpet==False:
                        return mild_nse
                    else:
                        return step_reward
                else:
                    if self.is_goal(state) == True:
                        return goal
                    else:
                        return step_reward
            return state_reward
