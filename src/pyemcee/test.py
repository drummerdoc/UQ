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
    def NumData(self):
        return self.d.NumData()
    def PriorMean(self):
        return self.d.PriorMean()
    def PriorStd(self):
        return self.d.PriorStd()
    def GenerateTestMeasurements(self, data):
        return self.d.GenerateTestMeasurements(data)

def lnprob(x, driver):
    """ Define the probability distribution that you would like to sample.

        Should be Log(P) based on parameters x
        Currently comes from driver object that wraps up all the c++ stuff
        that combines prior, the simulation result and the experimental data
        distribution to get the likelihood of the sample from the parameter
        space

    """
    driver.count += 1
    if driver.count % 1000 == 0:
        print("driver call number ", driver.count)
    return driver.Eval(x)

driver = DriverWrap()
driver.d = pymc.Driver(len(sys.argv), sys.argv)
driver.count = 0

ndim = driver.NumParams()
ndata = driver.NumData()
prior_mean = driver.PriorMean()
prior_std = driver.PriorStd()

print('Number of Parameters:',ndim)
print('Number of Data:',ndata)
print('Prior Mean/Std: ')
print(zip(prior_mean,prior_std))

nwalkers = 10
print('Number of Walkers:',nwalkers)

# Choose an initial set of positions for the walkers.
p0 = [prior_mean + np.random.rand(ndim) * prior_std for i in xrange(nwalkers)]

print('Initial walker parameters: ')
for walker in p0:
    print(walker)

# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[driver])

# Run burn-in steps
print ('Doing burn-in...')
pos, prob, state = sampler.run_mcmc(p0, 50)
#pos, prob, state = sampler.run_mcmc(p0, 10)
print ('Burn-in complete, number of evals:',driver.count)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, do sample steps.
print ('Sampling...')
#sampler.run_mcmc(pos, 20, rstate0=state)
#sampler.run_mcmc(pos, 200, rstate0=state)
sampler.run_mcmc(pos, 2000, rstate0=state)
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
for i in range(ndim):
    f=open('hist'+str(i)+'.dat','wb')
    pickle.dump(sampler.flatchain[:,i],f)
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

posterior_mean = []
for i in range(ndim):
    posterior_mean.append(sampler.flatchain[:,i].mean()) 
    print('New mean:',i,posterior_mean[i])

print('Sample at prior mean:',driver.GenerateTestMeasurements(prior_mean))
print('Sample at posterior mean:',driver.GenerateTestMeasurements(posterior_mean))
