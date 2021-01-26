import math, csv, time
from collections import defaultdict
from el_agent import ELAgent
from environment import Environment, Util, State, RewardCalcMethod


class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              render=False, report_interval=50):
        self.init_log()
        actions = list(range(len(env.points)))
        self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            env.reset()
            s = env.agent_state
            # Play until the end of episode.
            experience = []
            while env.can_action_at(s):
                a = self.policy(env)
                n_state, reward = env.step(a)
                experience.append({"state": s, "action": a, "reward": reward})
                s = n_state
            else:
                self.log(reward)

            # Evaluate each state, action.
            for i, x in enumerate(experience):
                #s, a = x["state"], x["action"]
                s = x["state"]
                a = Util.point2index(x["action"], env.points)

                # Calculate discounted future reward of s.
                G, t = Util.calc_reward_prev(env, s), 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                N[s][a] += 1  # count of s, a pair
                alpha = 1 / N[s][a]
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    # prepare
    #id = '214532'
    #id = '222911'
    #id = '316632'
    #id = '512848'
    id = '612825'

    #agent = MonteCarloAgent(epsilon=0.1)
    #agent = MonteCarloAgent(epsilon=0.2)
    #agent = MonteCarloAgent(epsilon=0.3)
    #agent = MonteCarloAgent(epsilon=0.4)
    #agent = MonteCarloAgent(epsilon=0.5)
    agent = MonteCarloAgent(epsilon=0.6)
    #agent = MonteCarloAgent(epsilon=0.7)
    #agent = MonteCarloAgent(epsilon=0.8)
    #agent = MonteCarloAgent(epsilon=0.9)

    points = Util.get_points(id)
    distance_matrix, time_matrix = Util.get_matrix(id)

    #env = Environment(points, distance_matrix, time_matrix, method=RewardCalcMethod.STRAIGHT, move_prob=1.0)
    #env = Environment(points, distance_matrix, time_matrix, method=RewardCalcMethod.DISTANCE, move_prob=1.0)
    env = Environment(points, distance_matrix, time_matrix, method=RewardCalcMethod.TIME, move_prob=1.0)

    time_sta = time.perf_counter()
    # learn
    agent.learn(env, episode_count=50000, gamma=1.0, report_interval=50)
    time_end = time.perf_counter()

    # output execution time
    print('learn time={0} [s]'.format(time_end - time_sta))
    #with open('../res/data/20200613/' + id + '/time_monte_carlo_straight.csv', mode='a') as f:
    #with open('../res/data/20200613/' + id + '/time_monte_carlo_distance.csv', mode='a') as f:
    with open('../res/data/20200613/' + id + '/time_monte_carlo_time.csv', mode='a') as f:
        f.write('{0}\n'.format(time_end - time_sta))

    # show result
    #agent.show_reward_log()
    best_state = Util.extract_best_state(env, agent.Q)
    Util.show_route(id, points, best_state.visited_points)


if __name__ == "__main__":
    train()



