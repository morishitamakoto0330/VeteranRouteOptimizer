import random
import csv
from environment import Environment

class Agent():

    def __init__(self, env):
        self.actions = env.actions

    def policy(self, env):
        return random.choice(env.actions)


def main():
    with open('sample_route.csv') as f:
        reader = csv.reader(f)
        points = [(float(row[0]), float(row[1])) for row in reader if len(row) != 0]

    print('sample route ==========================')
    print('{0} points'.format(len(points)))
    print('longitude, latitude')
    for p in points:
        print('({0}, {1})'.format(p[0], p[1]))

    env = Environment(points)
    agent = Agent(env)

    print('start ===============================')
    for i in range(10):
        state = env.reset()
        total_reward = 0

        while env.can_action_at(state):
            action = agent.policy(env)
            next_state, reward = env.step(action)
            total_reward += reward
            state = next_state

        print('Episode {0}: Agent gets {1} reward.'.format(i, total_reward))

    print('end =================================')


if __name__=="__main__":
    main()






