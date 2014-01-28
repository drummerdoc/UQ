from __future__ import print_function
import numpy as np
import emcee

import pyemcee as pymc

class DriverWrap:
    def Eval(self, data):
        return self.d.LogLikelihood(data)
    def NumParams(self):
        return self.d.NumParams()

# First, define the probability distribution that you would like to sample.
def lnprob(x, driver):
    driver.count += 1 
    return driver.Eval(x)

driver = DriverWrap()
driver.d = pymc.Driver()
driver.count = 0

ndim = driver.NumParams()
print('ndim:',ndim)
nwalkers = 10

# Choose an initial set of positions for the walkers.
p0 = [np.random.rand(ndim)*20+12000 for i in xrange(nwalkers)]

print('p0:',p0)

# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[driver])

# Run burn-in steps
pos, prob, state = sampler.run_mcmc(p0, 100)
print ('Burn-in complete, number of evals:',driver.count)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, do sample steps.
sampler.run_mcmc(pos, 10000, rstate0=state)
print ('Sampling complete, number of evals:',driver.count)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 250-dimensional
# vector.
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# If you have installed acor (http://github.com/dfm/acor), you can estimate
# the autocorrelation time for the chain. The autocorrelation time is also
# a vector with 10 entries (one for each dimension of parameter space).
#try:
#    print("Autocorrelation time:", sampler.acor)
#except ImportError:
#    print("You can install acor: http://github.com/dfm/acor")

import cPickle as pickle
f=open('hist.dat','wb')
pickle.dump(sampler.flatchain[:,0],f)
f.close()

# Finally, you can plot the projected histograms of the samples using
# matplotlib as follows (as long as you have it installed).
#try:
#    import matplotlib.pyplot as pl
#except ImportError:
#    print("Try installing matplotlib to generate some sweet plots...")
#else:
#    pl.hist(sampler.flatchain[:,0], 100)
#    pl.show()
    
