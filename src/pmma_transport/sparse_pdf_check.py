import sparse_pdf
import numpy as np
from is_tools import Resampling
import sparse_pdf
import triangle
import matplotlib.pyplot as plt


def spdf_weighted_samples(spdf, M=1000000):

    # hardcoded for 4D sparse pdf
    N = 4
    bins0 = spdf.get_bincens(0)
    bins1 = spdf.get_bincens(1)
    bins2 = spdf.get_bincens(2)
    bins3 = spdf.get_bincens(3)

    # Generate M uniformly distributed samples
    samples = np.zeros([N, M])
    for i in range(M):
        samples[0, i] = np.random.uniform(low=bins0[0], high=bins0[-1])
        samples[1, i] = np.random.uniform(low=bins1[0], high=bins1[-1])
        samples[2, i] = np.random.uniform(low=bins2[0], high=bins2[-1])
        samples[3, i] = np.random.uniform(low=bins3[0], high=bins3[-1])

    # Assign weights from sparse pdf
    w = np.zeros(M)
    for i in range(M):
        w[i] = spdf.getProbability(samples[:, i])

    wsum = np.sum(w)
    w = w/wsum
    print "sum of weights = ", np.sum(w)
    # Resample
    rs_map = Resampling(w, samples)

    samples_rw = samples[:, rs_map]

    return samples_rw


if __name__ == "__main__":
    # Draw samples from a 4d gaussian. make spdf. draw triangle plot
    M = 1000000
    N = 4
    spdf = sparse_pdf.sparsePdf(N)
    samples = np.zeros([N, M])
    mean = np.zeros(N)
    cov = np.ones([N, N])*0.5
    for i in range(M):
        # samples[:, i] = np.random.multivariate_normal(mean, cov)
        samples[:, i] = np.random.normal(size=4)

    print samples.shape
    fig = triangle.corner(samples.transpose(),
                          labels=["D0", "K0", "D1", "K1"])
    plt.savefig("input_samples.png")

    spdf.clear()
    lb = np.min(samples, axis=1)
    ub = np.max(samples, axis=1)
    print "lower bounds:", lb
    print "upper bounds:", ub
    spdf.setLowerBounds(lb)
    spdf.setUpperBounds(ub)
    counts = np.array([15, 15, 15, 15], dtype='int')
    spdf.setBinCounts(counts)
    for s in samples.transpose():
        spdf.addSamplePoint(s)
    spdf.normalize()

    spdf_samples = spdf_weighted_samples(spdf, M=1000000)

    fig = triangle.corner(spdf_samples.transpose(),
                          labels=["D0", "K0", "D1", "K1"])
    plt.savefig("spdf_samples.png")
