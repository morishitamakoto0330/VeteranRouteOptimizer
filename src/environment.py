import numpy as np


class State():

    def __init__(self, visited_points=[], lat=-1, lng=-1):
        self.lat = lat
        self.lng = lng
        self.visited_points = visited_points
        self.visited_points.append((lat, lng))

    def __repr__(self):
        return '<State: current_point=[{0}, {1}], visited_points_num={2}>'.format(self.lat, self.lng, len(self.visited_points))

    def clone(self):
        return State(self.visited_point[:-1], self.lat, self.lng)

    def __hash__(self):
        #return hash((self.lat, self.lng))
        return hash(tuple(self.visited_points))

    def __eq__(self, other):
        if self.lat != other.lat or self.lng != other.lng:
            return False
        if len(self.visited_points) != len(other.visited_points):
            return False
        else:
            for i in range(len(self.visited_points)):
                self_point = self.visited_points[i]
                other_point = other.visited_points[i]
                if self_point != other_point:
                    return False
        return True



class Environment():

    def __init__(self, points, move_prob=0.8):
        self.points = points
        self.agent_state = State()
        self.default_reward = -0.04
        self.move_prob = move_prob
        self.start_point = (-1, -1)
        self.reset()

    @property
    def lat_length(self):
        return len(self.points[0])

    @property
    def lng_length(self):
        return len(self.points[1])

    def actions(self, state):
        not_visited = []
        for p in self.points:
            if p not in state.visited_points:
                not_visited.append(p)

        return not_visited

    def states(self, state):
        states = []
        for (lat, lng) in self.points:
            if (lat, lng) not in state.visited_points:
                states.append(State(state.visited_points[:], lat, lng))
        return states

    def transit_func(self, state, action):
        transition_probs = {}

        if not self.can_action_at(state):
            return transition_probs

        if len(self.actions(state)) == 1:
            next_state = self._move(state, self.actions(state)[0])
            transition_probs[next_state] = 1.0
            return transition_probs

        for a in self.actions(state):
            prob = 0.0
            if a == action:
                prob = self.move_prob
            else:
                prob = (1.0 - self.move_prob)/(len(self.actions(state)) - 1)

            next_state = self._move(state, a)
            transition_probs[next_state] = prob

        return transition_probs

    def can_action_at(self, state):
        # Agent visited all point or not
        return len(state.visited_points) < len(self.points)

    def _move(self, state, action):
        if not self.can_action_at(state):
            return None

        next_state = State(state.visited_points[:], action[0], action[1])

        return next_state

    def reward_func(self, current_state, next_state):
        reward = self.default_reward
        _lat = current_state.lat - next_state.lat
        _lng = current_state.lng - next_state.lng

        if _lat == 0.0 and _lng == 0.0:
            reward -= 1
            #raise Exception('Agent has to move somewhere, but Agent is stopped.')
        else:
            reward += 1.0 / np.sqrt(_lat*_lat + _lng*_lng)

        if len(self.actions(current_state)) == 1:
            _lat = next_state.lat - self.start_point[0]
            _lng = next_state.lng - self.start_point[1]
            reward += 1.0 / np.sqrt(_lat*_lat + _lng*_lng)

        return reward

    def reset(self):
        # start point
        point = self.points[0]
        self.start_point = point
        # set initial value
        self.agent_state = State([], point[0], point[1])

        return self.agent_state

    def step(self, action):
        next_state, reward = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward

    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, 0

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)

        if state != next_state:
            reward = self.reward_func(state, next_state)
            # ato de yaru
            #self.visited_points.append((next_state.lon, next_state.lat))

            return next_state, reward

        else:
            return None, 0


















