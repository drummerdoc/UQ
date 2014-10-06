from mpi4py import MPI
import numpy as np
from H_MPIPool import H_MPIPool
import sys
import emcee
from emcee.utils import MPIPool


class DriverWrap:
    count = 0
    grank = -1
    wrank = -1
    comm = -1

    def __init__(self, comm):
        self.comm = comm
        self.grank = MPI.COMM_WORLD.Get_rank()
        self.wrank = self.comm.Get_rank()

    def Eval(self, data):
        self.comm.barrier()
        return -0.5 * np.sum(data**2)


# Setup hierarchical pool (e.g., each function evaluation
# gets a subcommunicator of ranks)
pool = H_MPIPool(MPI.COMM_WORLD)
pool.pass_communicator_to_function = False
pool.pass_argument_to_function = True

# This is the size of each communicator - with the
# first being a 1 for the master
pool.setSubrankCounts(distList=[1, 2, 4, 2])

# Could also just ask for them all the same size
# pool.setSubrankCounts(nTopRanks=3, nSubRanks=3)
# print "Subrank distribution: ", pool.sr


# Once that's specified, call a second init function
pool.setupPool()
# pool = MPIPool()

driver = DriverWrap(pool.worker_comm)

# This is a bit funky. Later, only the root (master) process calls
# the sampler, and the arguments passed there are pickled and
# sent to the workers with the function. But, we want
# every rank to have it's own "driver" instance which
# will contain the communicators, etc, it needs
# to maintain it's own parallel environment. This
# saves the driver from each rank and when the function
# is called in parallel the rank-specific driver function
# is inserted into the call
pool.argument = driver


def lnprob(x, ldriver):
    ldriver.count += 1
    if ldriver.count % 1000 == 0:
        print("this driver call number ", ldriver.count)
    return ldriver.Eval(x)

# emcee setup
ndim = 10
nwalkers = 20
p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]

if not pool.is_master():
    pool.wait()
    sys.exit(0)

# serial optoin
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob) #serial option

# note no driver argument - because no it comes from the
# pool in the parallel form
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
#
sampler.run_mcmc(pos, 100)

import matplotlib.pyplot as plt

if(pool.is_master()):
    plt.figure()
    plt.hist(sampler.flatchain[:, 1], 100, color="k", histtype="step")
    plt.title("Dimension {0:d}".format(1))
    plt.savefig("output_par.pdf", format='pdf')

pool.close()
