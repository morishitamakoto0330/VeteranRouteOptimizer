import csv
import numpy as np
import matplotlib.pyplot as plt

def show_route_data():
    points = []
    route = []

    print("sample data on 2020/6/13 =================================")
    with open('../../res/data/20200613/512848/points.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            lat = float(row[0])
            lng = float(row[1])
            point = (lat, lng)
            points.append(point)

    print("route data on 2020/6/13 =================================")
    with open('../../res/data/20200613/512848/episode=50000/epsilon=0.5/monte_carlo_straight.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            _route = []
            for r in row:
                if r != '':
                    _route.append(int(r))
            print(_route)
            route.append(_route)

    # plot route
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False)

    for i in range(2):
        for j in range(5):
            x, y = [], []
            for index in route[i*5 + j]:
                x.append(points[index][0])
                y.append(points[index][1])
            axes[i,j].xaxis.set_visible(False)
            axes[i,j].yaxis.set_visible(False)
            axes[i,j].plot(x, y, color='red')

    plt.show()
    




if __name__ == "__main__":
    show_route_data()


