import numpy as np


class State():

    def __init__(self, lon=-1, lat=-1):
        self.lon = lon
        self.lat = lat

    def __repr__(self):
        return '<State: [{0}, {1}]>'.format(self.lon, self.lat)

    def clone(self):
        return State(self.lon, self.lat)

    def __hash__(self):
        return hash((self.lon, self.lat))

    def __eq__(self, other):
        return self.lon == other.lon and self.lat == other.lat


class Action():
    pass


class Environment():

    def __init__(self, points, move_prob=0.8):
        self.points = points
        self.visited_points = []
        self.agent_state = State()
        self.default_reward = -0.04
        self.move_prob = move_prob
        self.reset()

    @property
    def lon_length(self):
        return len(self.points[0])

    @property
    def lat_length(self):
        return len(self.points[1])

    @property
    def actions(self):
        visited = self.visited_points
        not_visited = []

        for p in self.points:
            isVisited = False
            for v in visited:
                if p[0] == v[0] and p[1] == v[1]:
                    isVisited = True
            if not isVisited:
                not_visited.append(p)

        return not_visited

    @property
    def states(self):
        states = []
        for (lon, lat) in self.points:
            states.append(State(lon, lat))
        return states

    def transit_func(self, state, action):
        transition_probs = {}

        if not self.can_action_at(state):
            return transition_probs

        if len(self.actions) == 1:
            next_state = self._move(state, self.actions[0])
            transition_probs[next_state] = 1.0
            return transition_probs

        for a in self.actions:
            prob = 0.0
            if a == action:
                prob = self.move_prob
            else:
                prob = (1.0 - self.move_prob)/(len(self.actions) - 1)

            next_state = self._move(state, a)
            transition_probs[next_state] = prob

        return transition_probs

    def can_action_at(self, state):
        # Agent visited all point or not
        return len(self.visited_points) < len(self.points)

    def _move(self, state, action):
        if not self.can_action_at(state):
            return None

        next_state = State(action[0], action[1])

        return next_state

    def reward_func(self, current_state, next_state):
        reward = self.default_reward

        _lon = current_state.lon - next_state.lon
        _lat = current_state.lat - next_state.lat

        if _lon == 0.0 and _lat == 0.0:
            reward -= 1000
            #raise Exception('Agent has to move somewhere, but Agent is stopped.')
        else:
            reward += 1.0 / np.sqrt(_lon*_lon + _lat*_lat)

        return reward

    def reset(self):
        # start point
        point = self.points[0]
        # set initial value
        self.visited_points = []
        self.visited_points.append(point)
        self.agent_state.lon = point[0]
        self.agent_state.lat = point[1]

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

        #next_state = self._move(state, action)

        if state != next_state:
            reward = self.reward_func(state, next_state)
            self.visited_points.append((next_state.lon, next_state.lat))

            return next_state, reward

        else:
            return None, 0


















