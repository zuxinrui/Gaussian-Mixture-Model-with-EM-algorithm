import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# generate dataset:
def data(Sigma, Mu1, Mu2, k, N):
    global X
    global Mu
    global Expectations
    X = np.zeros((1, N))
    Mu = np.random.random(2)
    Expectations = np.zeros((N, k))
    for i in xrange(0, N):
        if np.random.random(1) > 0.5:
            X[0, i] = np.random.normal() * Sigma + Mu1
        else:
            X[0, i] = np.random.normal() * Sigma + Mu2

# e step:


def Estep(Sigma, k, N):
    global Expectations
    global Mu
    global X
    for i in xrange(0, N):
        Denom = 0
        for j in xrange(0, k):
            Denom += math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(X[0, i] - Mu[j])) ** 2)
        for j in xrange(0, k):
            Numer = math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(X[0, i] - Mu[j])) ** 2)
            Expectations[i, j] = Numer / Denom

# m step:


def Mstep(k, N):
    global Expectations
    global X
    for j in xrange(0, k):
        Numer = 0
        Denom = 0
        for i in xrange(0, N):
            Numer += Expectations[i, j] * X[0, i]
            Denom += Expectations[i, j]
        Mu[j] = Numer / Denom


def run(Sigma, Mu1, Mu2, k, N, iter_num, Epsilon):
    data(Sigma, Mu1, Mu2, k, N)
    print "<u1,u2>:", Mu
    for i in range(iter_num):
        Old_Mu = copy.deepcopy(Mu)
        Estep(Sigma, k, N)
        Mstep(k, N)
        print i, Mu
        if sum(abs(Mu - Old_Mu)) < Epsilon:
            break


if __name__ == '__main__':
    run(16, 60, -10, 2, 1000, 1000, 0.0001)
    plt.hist(X[0, :], 50)
    plt.show()