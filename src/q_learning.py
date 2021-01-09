import math
import csv
from collections import defaultdict
from el_agent import ELAgent
from environment import Environment, Util, State, RewardCalcMethod


class QLearningAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        actions = list(range(len(env.points)))
        self.Q = defaultdict(lambda: [0] * len(actions))
        for e in range(episode_count):
            env.reset()
            s = env.agent_state
            while env.can_action_at(s):
                a = self.policy(env)
                n_state, reward = env.step(a)

                gain = reward + gamma * max(self.Q[n_state])
                _a = Util.point2index(a, env.points)
                estimated = self.Q[s][_a]
                self.Q[s][_a] += learning_rate * (gain - estimated)
                s = n_state

            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    # prepare
    agent = QLearningAgent(epsilon=0.1)
    points = Util.get_points()
    distance_matrix, time_matrix = Util.get_matrix()

    env = Environment(points, distance_matrix, time_matrix, method=RewardCalcMethod.STRAIGHT, move_prob=1.0)
    #env = Environment(points, distance_matrix, time_matrix, method=RewardCalcMethod.DISTANCE, move_prob=1.0)
    #env = Environment(points, distance_matrix, time_matrix, method=RewardCalcMethod.TIME, move_prob=1.0)

    # learn
    agent.learn(env, episode_count=500, gamma=1.0, report_interval=50)
    agent.show_reward_log()

    # result
    best_state = Util.extract_best_state(env, agent.Q)
    Util.show_route(points, best_state.visited_points)

if __name__ == "__main__":
    train()




