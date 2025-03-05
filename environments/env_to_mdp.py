from ai_safety_gridworlds.environments.shared.safety_game import *

class EnvToMDP:
    def __init__(self, grid, is_oracle=False, is_lrtdp=True, calibration_reward=[]):
        self.env = grid
        self.grid = grid.last_observations['board']
        self._value_mapping = {0:'#', 1:'.', 2:'A', 3:'V', 4:'$', 5:'G'}
        self.grid_list = [[self._value_mapping[self.grid[i][j]] for j in range(len(self.grid[i]))] for i in range(len(self.grid))]
        self.num_actions = len(range(grid.action_spec().maximum + 1))
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.num_states = self.rows * self.cols
        self.num_states_not_wall = self.num_states - sum(row.count('#') for row in self.grid_list)
        self.agent_reward_cache = {}
        self.learned_reward_cache = {}
        self.oracle_demos = {}
        self.num_walls = sum(row.count('#') for row in self.grid_list)
        self.critical_state_preds = {}
        self.reward_cache = {}
        self.oracle_q_values = {}
        self.agent_q_values = {}
        self.is_oracle = is_oracle
        self.is_lrtdp = is_lrtdp
        self.calibration_reward = calibration_reward
        self.is_baseline = False
        self.get_all_reward = False
        self.all_states = None
        self.domain = 'vase'
        self.start_state = (((np.where(self.grid == 2)[0].item(), np.where(self.grid == 2)[1].item()), False, False))
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
                for action in range(self.num_actions):
                    self.oracle_q_values[(state, action)] = 0.001
                    self.agent_q_values[(state, action)] = 0.001
        self.terminal_state = (((np.where(self.grid == 5)[0].item(), np.where(self.grid == 5)[1].item()), False, False))
        self.state = (((np.where(self.grid == 2)[0].item(), np.where(self.grid == 2)[1].item()), False, False))
        self.all_states = self.getStateFactorRep()

    def getStateFactorRep(self):
        featureRep = []
        for i in range(self.rows):
            for j in range(self.cols):
                currState = self.grid_list[i][j]
                if currState == 'G':
                    self.terminal_state = ((i, j), False, False)
                    featureRep.append(((i, j), False, False))
                elif currState == 'V':
                    featureRep.append(((i, j), True, True))
                elif currState == '$':
                    featureRep.append(((i, j), True, False))
                elif currState != '#':
                    featureRep.append(((i, j), False, False))
        return featureRep

    def step(self, action, evaluate=False):
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
        else: # down
            return (1,0)

    def get_actions(self, state):
        return [0, 1, 2, 3]

    def is_boundary(self, state):
        x, y = state
        return (x <= 0 or x >= self.rows-1 or y <= 0 or y >= self.cols-1 )

    def is_goal(self, state):
        return state == self.terminal_state

    def move(self, currFactoredState, action):
        state, vase, carpet = currFactoredState
        new_state = tuple(x + y for (x, y) in zip(state, self.getActionFactorRep(action)))
        if self.is_boundary(new_state):
            # self.state = currFactoredState
            return currFactoredState, True
        else:
            if self.grid_list[new_state[0]][new_state[1]] == 'V':
                return (new_state, True, True), False
            elif self.grid_list[new_state[0]][new_state[1]] == '$':
                return (new_state, True, False), False
            else:
                return (new_state, False, False), False

    def get_side_states(self, state, action):
        side_states =[]
        for a in range(self.num_actions):
            if a != action:
                new_state, is_wall = self.move(state, a)
                if not is_wall:
                    side_states.append(new_state)
                elif is_wall:
                    side_states.append(state)
        return side_states

    def get_transition(self, curr_state, action, next_state):
        succ_factored_state, is_wall = self.move(curr_state, action)
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
        return 0

    def get_possible_next_states(self, state):
        possible_states = set()
        for action in range(self.num_actions):
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

    def get_reward(self, state, action, is_oracle=True, is_lrtdp=None):
        (x,y), vase, carpet = state
        nse = None
        is_lrtdp = self.is_lrtdp

        if is_lrtdp==True:
            goal = 0
            severe_nse = 10
            mild_nse = 5
            if len(self.calibration_reward)!=0:
                nse=self.calibration_reward
            step = 1
        state_reward = None
        if self.is_baseline==True:
            if self.is_goal(state) == True:
                return goal
            else:
                r = step
                r += (nse[0]*vase) + (nse[1]*carpet)
                return r
        else:
            if action in [0,1,2,3]:
                if is_oracle or self.get_all_reward:
                    if self.is_goal(state) == True:
                        return goal
                    elif vase==True and carpet==False:
                        return severe_nse
                    elif vase==True and carpet==True:
                        return mild_nse
                    else:
                        return step
                else:
                    if self.is_goal(state) == True:
                        return goal
                    else:
                        return step
            return state_reward
