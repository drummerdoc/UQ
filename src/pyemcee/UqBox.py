#
#  UQBox code (pronounced like "jukebox" without the "j")
#
# chief software engineer: Marcus Day
# helper elf programmers:  Jonathan Goodman (goodman@cims.nyu.edu)
#                          John Bell
#                          Matthias Morzfeld

from __future__ import print_function
import numpy as np
import emcee
import sys

import pyemcee as pymc
import cPickle

#   Inputs
nwalkers      = 10
nBurnIn       = 10         # Number of burn-in samples before starting to take data
nChainLength  = 5000       # Number of MCMC resamplings in the data run, after burn in
outFilePrefix = "Results_" # Prefix to output file names, to be appended with eval #
outFilePeriod = 5000       # Number of samples between calls to write data
runlogPeriod  = 1000       # Number of samples between info messages written to screen
nDigits       = int(np.log10( nwalkers*(1+nBurnIn+nChainLength))) + 1 # Number of digits in appended number

# Pickle entire sample chain (cummulative, and therefore not exactly ideal, but simple)
def PickleResults(driver,filename):
    print("Writing output: "+filename)
    x = sampler.chain
    OutNames = ['x',    # The names here must be the exact variable names
                'ndim',
                'ndata',
                'nwalkers',
                'nBurnIn',
                'nChainLength']

    OutDict = dict()     # a python dictionary with variable names and values
    for name in OutNames:
        CommString = "OutDict['" + name + "']  =  " + name  # produce a command like:  OutDict['x'] = x
        exec CommString

    # Then pickle the dictionary
    ResultsFile = open( filename, 'wb')
    Output      = [ OutNames, OutDict]
    cPickle.dump( Output, ResultsFile)
    ResultsFile.close()

#
# Simple driver to enable persistent static class wrapped around driver object
#
# Construction of the contained "Driver" will read the input file listed on the
#  command line, set up the active parameters and synthetic experiments described
#  therein.
#
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
    def EnsembleStd(self):
        return self.d.EnsembleStd()
    def GenerateTestMeasurements(self, data):
        return self.d.GenerateTestMeasurements(data)

#
# The function called by emcee to sample the posterior
#
# The key job is to orchestrate the corresponding Eval call with a vector
#  of sampled values of the parameters. Also manage periodic dump of info/data.
# 
def lnprob(x, driver):
    """ Define the probability distribution that you would like to sample.

        Should be Log(P) based on parameters x
        Currently comes from driver object that wraps up all the c++ stuff
        that combines prior, the simulation result and the experimental data
        distribution to get the likelihood of the sample from the parameter
        space

    """
    driver.count += 1
    if driver.count % runlogPeriod == 0:
        print("driver called ", driver.count, "times")

    if driver.count % outFilePeriod == 0:
        fmt = "%0"+str(nDigits)+"d"
        outFileName = outFilePrefix + (fmt % driver.count)
        PickleResults(sampler,outFileName)
        
    return driver.Eval(x)

# Build the persistent class containing the driver object
driver = DriverWrap()
driver.d = pymc.Driver(len(sys.argv), sys.argv)
driver.count = 0

ndim = driver.NumParams()
ndata = driver.NumData()
prior_mean = driver.PriorMean()
prior_std = driver.PriorStd()
ensemble_std = driver.EnsembleStd()

print('Number of Parameters:',ndim)
print('Number of Data:',ndata)
print('prior means:  '+ str(prior_mean))
print('prior std: '+ str(prior_std))
print('ensemble std: '+ str(ensemble_std))

# Choose an initial set of positions for the walkers.
p0 = [prior_mean + np.random.rand(ndim) * ensemble_std for i in xrange(nwalkers)]

print('Initial walker parameters: ')
for walker in p0:
    print(walker)

# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[driver])
driver.sampler = sampler

# Run burn-in steps
print ('Doing burn-in...')
pos, prob, state = sampler.run_mcmc(p0, nBurnIn)
#pos, prob, state = sampler.run_mcmc(p0, 10)
print ('Burn-in complete, number of evals:',driver.count)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, do sample steps.
print ('Sampling...')
#sampler.run_mcmc(pos, 20, rstate0=state)
#sampler.run_mcmc(pos, 200, rstate0=state)
sampler.run_mcmc(pos, nChainLength, rstate0=state)
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

fmt = "%0"+str(nDigits)+"d"
outFileName = outFilePrefix + (fmt % driver.count) + ".dat"
PickleResults(sampler,outFileName)

posterior_mean = []
for i in range(ndim):
    posterior_mean.append(sampler.flatchain[:,i].mean()) 
    print('New mean:',i,posterior_mean[i])

