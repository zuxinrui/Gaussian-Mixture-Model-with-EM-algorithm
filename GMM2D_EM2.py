import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import seaborn

colors = ['red', 'turquoise', 'orange', 'blue', 'navy', 'black']


def data2d(sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, N, mu1, mu2, mu3, mu4, mu5, mu6, alpha):
    global X
    X = np.zeros((N, 2))
    X = np.matrix(X)
    global mu
    mu = np.random.random((6, 2))
    mu = np.matrix(mu)
    global excep
    excep = np.zeros((N, 6))
    global alpha_
    alpha_ = [0.16, 0.16, 0.2, 0.16, 0.16, 0.16]
    for i in range(N):
        if np.random.random(1) < 0.1:
            X[i, :] = np.random.multivariate_normal(mu1, sigma1, 1)
        elif 0.1 <= np.random.random(1) < 0.3:
            X[i, :] = np.random.multivariate_normal(mu2, sigma2, 1)
        elif 0.3 <= np.random.random(1) < 0.5:
            X[i, :] = np.random.multivariate_normal(mu3, sigma3, 1)
        elif 0.5 <= np.random.random(1) < 0.7:
            X[i, :] = np.random.multivariate_normal(mu4, sigma4, 1)
        elif 0.7 <= np.random.random(1) < 0.9:
            X[i, :] = np.random.multivariate_normal(mu5, sigma5, 1)
        else:
            X[i, :] = np.random.multivariate_normal(mu6, sigma6, 1)
    print 'x:', X
    print 'mu:', mu


def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        covariances = gmm.covariances_[n][:2, :2]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


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


def fixgmm():
    gmm = mixture.GaussianMixture(n_components=6, covariance_type='full', max_iter=40, random_state=0)
    gmm.fit(X)
    print '***covariance:***', gmm.covariances_
    print '***mean:***', gmm.means_
    print '***weight:***', gmm.weights_
    y_pred = gmm.predict(X)
    colors = ['red', 'turquoise', 'orange', 'blue', 'navy', 'black']
    for n, color in enumerate(colors):
        data = X[y_pred == n]
        plt.scatter(data[:, 0], data[:, 1], marker='o', s=25, color=color, alpha=0.4)


if __name__ == '__main__':
    iter_num = 2000
    N = 5000
    k = 6
    probility = np.zeros(N)
    u1 = [5, 45]
    u2 = [25, 40]
    u3 = [20, 10]
    u4 = [45, 25]
    u5 = [50, 55]
    u6 = [22, 80]
    s1 = np.matrix([[30, 20], [20, 30]])
    s2 = np.matrix([[25, 5], [5, 40]])
    s3 = np.matrix([[35, -25], [-25, 25]])
    s4 = np.matrix([[20, -10], [-10, 30]])
    s5 = np.matrix([[40, -30], [-30, 30]])
    s6 = np.matrix([[30, 5], [5, 30]])
    alpha = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
    data2d(s1, s2, s3, s4, s5, s6, N, u1, u2, u3, u4, u5, u6, alpha)
    # plt.scatter(X[:, 0], X[:, 1], c='b', s=25, alpha=0.4, marker='o')
    fixgmm()
    plt.show()