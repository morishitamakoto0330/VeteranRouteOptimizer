import random
import numpy as np
import csv
from environment import Environment, State


class EpsilonGreedyAgent():

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.V = []

    def policy(self):
        coins = range(len(self.V))
        if random.random() < self.epsilon:
            return random.choice(coins)
        else:
            return np.argmax(self.V)

    def play(self, env):
        # Initialize estimation.
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1
            self.V[selected_coin] = new_average

        return rewards


if __name__ == "__main__":
	import pandas as pd
	import matplotlib.pyplot as plt

	def main():
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
					print("({0},{1})".format(lat, lng))
		# environment
		env = Environment(points, move_prob=1.0)


	main()

