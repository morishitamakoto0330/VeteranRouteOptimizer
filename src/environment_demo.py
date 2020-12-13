import random
import csv
from environment import Environment

class Agent():

    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        return random.choice(self.actions)


def main():
    with open('sample_route.csv') as f:
        reader = csv.reader(f)
        points = [(float(row[0]), float(row[1])) for row in reader if len(row) != 0]

    print('sample route ==========================')
    print('longitude, latitude')
    for p in points:
        print('({0}, {1})'.format(p[0], p[1]))

    env = Environment(points)
    agent = Agent(env)

    print('start ===============================')
    for i in range(10):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print('Episode {0}: Agent gets {1} reward.'.format(i, total_reward))

    print('end =================================')


if __name__=="__main__":
    main()






