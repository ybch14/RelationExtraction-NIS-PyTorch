#!/usr/bin/python3.6
#encoding=utf-8
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

color = ['b', 'g',  'c', 'r', 'm', 'y', 'k', 'gray']

def getXY(file_name, n=2000, sample=1):
    x, y = [], []
    with open(file_name) as f:
        for i, line in enumerate(f):
            if i % sample == 0:
                entries = line.split()
                y.append(float(entries[0]))
                x.append(float(entries[1]))
                if i >= n:
                    break
    return x, y

def plot_mul(files, colors, output, legends=None, markers=None, linestyle=None):
    assert(len(files) <= len(colors))
    for i, f in enumerate(files):
        path = f  #'.format(f)
        if not os.path.exists(path):
            print('{} does not exists'.format(f))
            continue
        x, y = getXY(path, n=2000, sample=1)
        if markers is None and linestyle is not None:
            plt.plot(x,y, color=colors[i], linestyle=linestyle[i])
        elif markers is not None and linestyle is not None:
            plt.plot(x,y, marker=markers[i], markevery=100, markersize=5, color=colors[i], linewidth=1.25, fillstyle='none', linestyle=linestyle[i])
        elif markers is None and linestyle is None:
            plt.plot(x,y, color=colors[i])
    if legends is not None:
        assert(len(legends) >= len(files))
        plt.legend(legends, prop={'size':12})
    plt.ylim([0.35, 1])
    plt.xlim([0.0, 0.45])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.gca().tick_params(labelsize=10)
    plt.grid(linestyle='--')
    # plt.show()
    plt.savefig(output, dpi=200)

if __name__ == "__main__":
    l = list(range(1, 11))
    plot_mul(["results/prfile/%d_pr.txt" % i for i in l], colors=[color[(i-1)%len(color)] for i in l], output="result.png", legends=[str(i) for i in l])
