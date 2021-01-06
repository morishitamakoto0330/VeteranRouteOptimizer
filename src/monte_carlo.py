import math
import csv
from collections import defaultdict
from el_agent import ELAgent
from environment import Environment, Util, State

def getPoints():
    # read file
    points = []
    print("sample route on 2020/6/13 =================================")
    with open('../res/data/20200613_512848.csv') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index != 0:
                lat = float(row[6])
                lng = float(row[7])
                points.append((lat, lng))
                print("{0}: ({1},{2})".format(index, lat, lng))
    return points

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
                G, t = 0, 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                N[s][a] += 1  # count of s, a pair
                alpha = 1 / N[s][a]
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    agent = MonteCarloAgent(epsilon=0.1)
    env = Environment(getPoints(), move_prob=1.0)
    agent.learn(env, episode_count=500, gamma=1.0, report_interval=50)
    agent.show_reward_log()

    ok = 0
    ng = 0
    for s in agent.Q:
        count = 0
        for q in agent.Q[s]:
            if q != 0:
                count += 1

        if count != 1:
            ok += 1
        else:
            ng += 1

    print('{0}: {1} vs. {2}'.format(len(agent.Q), ok, ng))


if __name__ == "__main__":
    train()
