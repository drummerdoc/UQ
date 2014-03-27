from __future__ import print_function
import numpy as np
import emcee
import sys

import pyemcee as pymc


class DriverWrap:
    def Eval(self, data):
        return self.d.LogLikelihood(data)

    def NumParams(self):
        return self.d.NumParams()


def lnprob(x, driver):
    """ Define the probability distribution that you would like to sample.

        Should be Log(P) based on parameters x
        Currently comes from driver object that wraps up all the c++ stuff
        that combines prior, the simulation result and the experimental data
        distribution to get the likelihood of the sample from the parameter
        space

    """
    driver.count += 1
    print "driver call number ", driver.count
    return driver.Eval(x)


# Set up driver object with reference to class from c++ infrastructure
driver = DriverWrap()
driver.d = pymc.Driver(len(sys.argv), sys.argv)
driver.count = 0

# Trial evaluation - do it 3 times
#x = (173400000.0, 671.0)
#for ii in xrange(3):
#    driver.Eval(x)

# Size of parameter space - set up through input file
ndim = driver.NumParams()
print('ndim:', ndim)
nwalkers = 10

p = np.array([173400000.0, 671.0])
p0 = []
for i in xrange(nwalkers):
    p0.append(p)
# Choose an initial set of positions in parameter space for the walkers.
# p0 = [np.random.rand(ndim)*20+12000 for i in xrange(nwalkers)]

#driver.Eval(p0)


print('Initial walker positions in parameter space (p0):', p0)

# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[driver])

# Run burn-in steps
pos, prob, state = sampler.run_mcmc(p0, 50)
print ('Burn-in complete, number of evals:', driver.count)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, do sample steps.
sampler.run_mcmc(pos, 2000, rstate0=state)
print ('Sampling complete, number of driver evaluations:', driver.count)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 250-dimensional
# vector.
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# If you have installed acor (http://github.com/dfm/acor), you can estimate
# the autocorrelation time for the chain. The autocorrelation time is also
# a vector with 10 entries (one for each dimension of parameter space).

# try:
#     print("Autocorrelation time:", sampler.acor)
# except ImportError:
#     print("You can install acor: http://github.com/dfm/acor")

# Serialize and save out histogram
import cPickle as pickle
f = open('hist.dat', 'wb')
pickle.dump(sampler.flatchain[:, 0], f)
f.close()

# Finally, you can plot the projected histograms of the samples using
# matplotlib as follows (as long as you have it installed).
try:
    import matplotlib.pyplot as pl
except ImportError:
    print("Try installing matplotlib to generate some sweet plots...")
else:
    pl.hist(sampler.flatchain[:,0], 100)
    pl.show()
