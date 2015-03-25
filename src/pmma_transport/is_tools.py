
from numpy import linalg as LA
import numpy as np


#  Implicit sampling routine
def Fo(x, phi, mu, evecs, evals):
    y = np.multiply(1 / np.sqrt(evals).T, evecs.T * (x-mu))
    return phi + 0.5 * (np.linalg.norm(y)**2)


def Fo2(x, phi, mu, L):
    y = np.linalg.solve(L, (x-mu))
    return phi + 0.5 * (np.linalg.norm(y)**2)


def CompR(w):
    N = len(w)
    w2mean = 0
    wmean = 0
    for i in range(N):
        wmean += w[i]
        w2mean += w[i]*w[i]
    wmean *= 1./N
    w2mean *= 1./N
    return w2mean/(wmean*wmean)


def WeightedMean(w, samples):
    N = samples.shape[0]
    M = samples.shape[1]
    CondMean = np.zeros(N)
    for n in range(N):
        for m in range(M):
            CondMean[n] += w[m]*samples[n, m]
    return CondMean


# This is from covModel.py
def Resampling(w, samples):
    N = samples.shape[0]
    M = samples.shape[1]
    c = np.zeros(M+1)
    for j in range(1, len(c)):
        c[j] = c[j-1] + w[j-1]
    i = 0
    u1 = np.random.rand(1)/M
    u = 0
    rs_map = np.zeros(M, dtype=int)
    for m in range(M):
        u = u1 + float(m)/M
        while u >= c[i]:
            i += 1
        rs_map[m] = i-1 # Note: i is never 0 here
    return rs_map

def WeightedVar(CondMean, w, samples):
    N = samples.shape[0]
    M = samples.shape[1]
    CondVar = np.zeros(N)
    for n in range(N):
        for m in range(M):
            CondVar[n] += w[m] * (samples[n, m] - CondMean[n])\
                               * (samples[n, m] - CondMean[n])
    return CondVar


def EffSampleSize(w):
    n = len(w)
    sumSq = 0
    for i in range(n):
        sumSq += w[i]*w[i]
    if sumSq == 0:
        return 0
    return 1/sumSq


def get_H_samples(initialSamples, N, M, scales):
    x = initialSamples[-M:, :N]  # Independent data
    z = np.matrix(initialSamples[-M:, -1])  # Dependent data, as column vector

    scaled_x = np.zeros_like(x)
    for i in range(M):
        scaled_x[i] = x[i]/scales

    Hinv = np.cov(scaled_x.T)
    return Hinv


def computeHEV(Hinv, neff):
    evals, evecs = LA.eigh(Hinv)
    evecs = np.matrix(evecs)
    sl = np.argsort(evals)
    evals = evals[sl]
    evecs = evecs[:, sl]

    print 'Eigenvalues:', evals
    evals = np.matrix(evals[-neff:])
    evecs = evecs[:, -neff:]
    print 'Eigenvalues kept:', evals
    return evecs, evals


def do_is(NOS, N, M, evecs, evals, mu, phi, data, lb, ub, neff,
          scales, lnlikefcn):

    Samples = np.matrix(np.zeros(shape=(N, NOS)))
    w = np.array(np.zeros(NOS))
    newF = np.array(np.zeros(NOS))
    F0 = np.array(np.zeros(NOS))

    print "Starting implicit sampling"
    for i in range(NOS):
        sample_oob = True
        while sample_oob:
            Samples[:, i] = mu + evecs*np.multiply(np.sqrt(evals),
                                                   np.random.randn(1, neff)).T
            sample_good = True
            for n in range(N):
                sample_good &= Samples[n, i] >= lb[n]\
                    and Samples[n, i] <= ub[n]
            sample_oob = not sample_good

        F0[i] = Fo(Samples[:, i], phi, mu, evecs, evals)
        Samples[:, i] = np.multiply(Samples[:, i].T, np.matrix(scales)).T
        xx = np.array(Samples[:, i].T)[0]
        newF[i] = lnlikefcn(xx, data)
        print "newF = ", newF[i]
        if newF[i] == np.inf:
            w[i] = -1
        else:
            w[i] = F0[i] - newF[i]

        print "Sample ", i, "of", NOS, "F0 =", F0[i], " F =", newF[i],\
              "w=", w[i]

    wmax = np.amax(w)
    plt.close()
    plt.plot(w)
    plt.show()
    print "wmax = ", wmax
    for i in range(NOS):
        if w[i] < 0:
            w[i] = 0
        else:
            w[i] = np.exp(w[i] - wmax)

    plt.close()
    plt.plot(w)
    plt.show()
    wsum = np.sum(w)
    w = w/wsum
    return w, Samples
