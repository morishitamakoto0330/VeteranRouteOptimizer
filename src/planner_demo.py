import csv
import numpy as np
from environment import Environment, State
from planner import ValueIterationPlanner
from matplotlib import pyplot as plt

def main():
    # read file
    with open('sample_route.csv') as f:
        reader = csv.reader(f)
        points = [(float(row[0]), float(row[1])) for row in reader if len(row) != 0]

    # print route
    print('sample route ==========================')
    print('{0} points'.format(len(points)))
    print('longitude, latitude')
    for p in points:
        print('({0}, {1})'.format(p[0], p[1]))

    # prepare environment 
    env = Environment(points)

    order = []
    for i in range(len(points) - 1):
        # value base plan
        planner = ValueIterationPlanner(env)
        result = planner.plan()
        print('{0}:'.format(i))
        print(result)

        # move next point
        next_point = points[result.index(max(result))]
        env.agent_state = State(next_point[0], next_point[1])
        env.visited_points.append(next_point)

        # save visit order
        order.append(next_point)

    # plot result
    x = []
    y = []
    for _x, _y in order:
        x.append(_x)
        y.append(_y)
    plt.scatter(x, y, c='red')
    plt.plot(x, y, c='red')
    plt.show()


if __name__=="__main__":
    main()




