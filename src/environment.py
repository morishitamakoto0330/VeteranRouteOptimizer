import numpy as np
import csv
import math
from matplotlib import pyplot as plt
from enum import Enum


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

class RewardCalcMethod(Enum):
    STRAIGHT = 1
    DISTANCE = 2
    TIME = 3

class Util():

    @staticmethod
    def get_points(id):
        points = []
        print("sample route of ID={0} on 2020/6/13 =================================".format(id))
        #with open('../res/data/20200613/512848/20200613_512848.csv') as f:
        with open('../res/data/20200613/' + id + '/20200613_' + id + '.csv') as f:
            reader = csv.reader(f)
            for index, row in enumerate(reader):
                if index != 0:
                    lat = float(row[6])
                    lng = float(row[7])
                    points.append((lat, lng))
                    print("{0}: ({1},{2})".format(index, lat, lng))
        print("")

        return points

    @staticmethod
    def get_matrix(id):
        distance_matrix = []
        time_matrix = []

        print("distance matrix of ID={0} on 2020/6/13 =================================".format(id))
        with open('../res/data/20200613/' + id + '/20200613_' + id + '_distance.csv') as f:
            reader = csv.reader(f)
            str = ''
            for row in reader:
                _distance_matrix = []
                for d in row:
                    _distance_matrix.append(float(d))
                    str += '{0:>10}'.format(float(d))
                distance_matrix.append(_distance_matrix)
                str += '\n'
            print(str)

        print("time matrix of ID={0} on 2020/6/13 =================================".format(id))
        with open('../res/data/20200613/' + id + '/20200613_' + id + '_duration.csv') as f:
            reader = csv.reader(f)
            str = ''
            for row in reader:
                _time_matrix = []
                for d in row:
                    _time_matrix.append(float(d))
                    str += '{0:>10} '.format(float(d))
                time_matrix.append(_time_matrix)
                str += '\n'
            print(str)

        return distance_matrix, time_matrix

    @staticmethod
    def point2index(point, points):
        for index, p in enumerate(points):
            if p==point:
                return index
        return -1

    @staticmethod
    def calc_reward(env, current_state, next_state):
        reward = 0.0

        current_p = (current_state.lat, current_state.lng)
        current_index = Util.point2index(current_p, env.points)
        next_p = (next_state.lat, next_state.lng)
        next_index = Util.point2index(next_p, env.points)

        if env.method == RewardCalcMethod.STRAIGHT:
            lat = current_state.lat - next_state.lat
            lng = current_state.lng - next_state.lng
            reward += 1.0 / np.sqrt(lat*lat + lng*lng)
        elif env.method == RewardCalcMethod.DISTANCE:
            reward += 1.0 / float(env.distance_matrix[current_index][next_index])
        elif env.method == RewardCalcMethod.TIME:
            reward += 1.0 / float(env.time_matrix[current_index][next_index])
        else:
            raise Exception('You specified invalid RewardCalcMethod.')

        return reward

    @staticmethod
    def calc_reward_prev(env, state):
        reward = 0.0
        points = state.visited_points

        if env.method == RewardCalcMethod.STRAIGHT:
            for i in range(len(points) - 1):
                lat = points[i][0] - points[i+1][0]
                lng = points[i][1] - points[i+1][1]
                reward += 1.0 / np.sqrt(lat*lat + lng*lng)
        elif env.method == RewardCalcMethod.DISTANCE:
            for i in range(len(points) - 1):
                current_index = Util.point2index(points[i], env.points)
                next_index = Util.point2index(points[i+1], env.points)
                reward += 1.0 / float(env.distance_matrix[current_index][next_index])
        elif env.method == RewardCalcMethod.TIME:
            for i in range(len(points) - 1):
                current_index = Util.point2index(points[i], env.points)
                next_index = Util.point2index(points[i+1], env.points)
                reward += 1.0 / float(env.time_matrix[current_index][next_index])
        else:
            raise Exception('You specified invalid RewardCalcMethod.')

        return reward

    @staticmethod
    def extract_best_state(env, Q):
        only_one_action = 0
        best_state = State()
        best_q = 0.0
        for s in Q:
            q = Util.calc_reward_prev(env, s)
            if best_q < q:
                best_q = q
                best_state = s
            count = 0
            for q in Q[s]:
                if q != 0:
                    count += 1
            if count == 1:
                only_one_action += 1
        print('Number of State = {0}({1}%)'.format(len(Q), len(Q) / math.factorial(len(env.points) - 1) * 100))
        print('Number of State (only one action) = {0}'.format(only_one_action))
        print('Best State: {0}'.format(best_state))
        print('Best Q: {0}'.format(best_q))

        return best_state

    @staticmethod
    def show_route(id, points, visited_points):
        x = []
        y = []
        for lat, lng in points:
            x.append(lat)
            y.append(lng)
        plt.scatter(x, y, c='red')

        x = []
        y = []
        for lat, lng in visited_points:
            x.append(lat)
            y.append(lng)

        # last visit point
        for p in points:
            if p not in visited_points:
                x.append(p[0])
                y.append(p[1])
                break

        # back start point
        x.append(points[0][0])
        y.append(points[0][1])

        # output route to file
        #with open('../res/data/20200613/' + id + '/monte_carlo_straight.csv', mode='a') as f:
        with open('../res/data/20200613/' + id + '/monte_carlo_distance.csv', mode='a') as f:
        #with open('../res/data/20200613/' + id + '/monte_carlo_time.csv', mode='a') as f:
            str = ''
            for i in range(len(x)):
                index = Util.point2index((x[i], y[i]), points)
                str += '{0},'.format(index)
            str += '\n'
            f.write(str)

        # plot route
        plt.plot(x, y, c='red')
        #plt.show()



class Environment():

    def __init__(self, points, distance_matrix, time_matrix, method=RewardCalcMethod.STRAIGHT, move_prob=1.0):
        self.points = points
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.method = method
        self.agent_state = State()
        self.default_reward = 0.0
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
                #prob = (1.0 - self.move_prob)/(len(self.actions(state)) - 1)
                prob = 0.0

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
        reward = Util.calc_reward(self, current_state, next_state)

        if len(self.actions(current_state)) == 1:
            next_index = Util.point2index((next_state.lat, next_state.lng), self.points)

            if self.method == RewardCalcMethod.STRAIGHT:
                lat = next_state.lat - self.start_point[0]
                lng = next_state.lng - self.start_point[1]
                reward += 1.0 / np.sqrt(lat*lat + lng*lng)
            elif self.method == RewardCalcMethod.DISTANCE:
                reward += 1.0 / float(self.distance_matrix[next_index][0])
            elif self.method == RewardCalcMethod.TIME:
                reward += 1.0 / float(self.time_matrix[next_index][0])
            else:
                raise Exception('You specified invalid RewardCalcMethod.')

        return reward

    def reset(self):
        # start point
        point = self.points[0]
        self.start_point = point
        # set initial value
        self.agent_state = State([], point[0], point[1])

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

            return next_state, reward

        else:
            return None, 0


















