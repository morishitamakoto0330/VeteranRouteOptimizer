import csv
from environment import Environment
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
    lon_max = 0.0
    lon_min = 1000.0
    lat_max = 0.0
    lat_min = 1000.0

    for p in points:
        if p[0] > lon_max:
            lon_max = p[0]
        if p[0] < lon_min:
            lon_min = p[0]
        if p[1] > lat_max:
            lat_max = p[1]
        if p[1] < lat_min:
            lat_min = p[1]
        print('({0}, {1})'.format(p[0], p[1]))

    print('normalization =========================')
    x = []
    y = []
    for p in points:
        lon = p[0]/lon_max
        lat = p[1]/lat_max
        print('({0}, {1})'.format(lon, lat))
        x.append(lat)
        y.append(lon)

    # value base plan
    env = Environment(points)
    planner = ValueIterationPlanner(env)
    result = planner.plan()

    # print result
    print(result)
    order = []
    for r in result:
        _order = len(result)
        for _r in result:
            if r > _r:
                _order -= 1
        order.append(_order)
    print(order)

    # plot result
    for i in range(len(result)):
        plt.scatter(x[i], y[i], label=str(order[i]))
    plt.legend(loc='upper left')
    plt.show()


if __name__=="__main__":
    main()




