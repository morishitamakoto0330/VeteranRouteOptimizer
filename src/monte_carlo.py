import math
import csv
from collections import defaultdict
from el_agent import ELAgent
from environment import Environment, Util, State


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
                G, t = Util.calc_reward(s), 0
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
    agent = MonteCarloAgent(epsilon=0.1)
    points = Util.get_points()
    d_matrix, t_matrix = Util.get_matrix()
    env = Environment(points, move_prob=1.0)

    # learn
    agent.learn(env, episode_count=500, gamma=1.0, report_interval=50)
    agent.show_reward_log()

    # result
    best_state = Util.extract_best_state(agent.Q)
    Util.show_route(points, best_state.visited_points)


if __name__ == "__main__":
    train()



