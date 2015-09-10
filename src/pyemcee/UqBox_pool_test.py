import numpy as np
import emcee
import sys
import matplotlib.pyplot as plt

from mpi4py import MPI

#from emcee.utils import MPIPool
from UqBox_pool import UqBoxPool

rank = MPI.COMM_WORLD.Get_rank()


def lnprob(x):
    return -0.5 * np.sum(x ** 2)

fcnarg = rank

def argfcn(x):
    return x

ndim, nwalkers = 10, 100
p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]


pool = UqBoxPool(MPI=MPI, debug=False)
pool.set_function(lnprob)
#pool.set_function_arg(fcnarg)

if not pool.is_master():
    pool.wait()
    sys.exit(0)

sampler = emcee.EnsembleSampler(nwalkers, ndim, argfcn, pool=pool)

nSteps = 400
sampler.run_mcmc(p0, nSteps)

plt.plot(sampler.chain[:,:,0])
plt.show()

pool.close()

