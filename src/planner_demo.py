import csv
import numpy as np
from environment import Environment, State
from planner import ValueIterationPlanner
from matplotlib import pyplot as plt

def main():
    # read file
    #with open('sample_route.csv') as f:
    with open('sample_route_mini.csv') as f:
        reader = csv.reader(f)

        points = []
        for row in reader:
            if len(row) != 0 and row[0] != '' and row[1] != '':
                lat = float(row[0].replace('"', ''))
                lng = float(row[1].replace('"', ''))
                points.append((lat, lng))

    # print route
    print('sample route ==========================')
    print('{0} points'.format(len(points)))
    print('longitude, latitude')
    for p in points:
        print('({0}, {1})'.format(p[0], p[1]))

    env = Environment(points, move_prob=1.0)
    planner = ValueIterationPlanner(env)
    result = planner.plan(gamma=1.0, threshold=0.0001)

    print(result)

    # plot result
    x = []
    y = []
    for _x, _y in points:
        x.append(_x)
        y.append(_y)
    plt.scatter(x, y, c='red')

    x = []
    y = []
    for _x, _y in result:
        x.append(_x)
        y.append(_y)
    x.append(x[0])
    y.append(y[0])

    plt.plot(x, y, c='red')
    plt.show()



if __name__=="__main__":
    main()




