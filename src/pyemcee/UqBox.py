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
from mpi4py import MPI


#   Inputs
nwalkers      = 62
nBurnIn       = 10        # Number of burn-in samples before starting to take data
nChainLength  = 5000       # Number of MCMC resamplings in the data run, after burn in
outFilePrefix = "Results_" # Prefix to output file names, to be appended with eval #
outFilePeriod = 500        # Number of samples between calls to write data
runlogPeriod  = 100        # Number of samples between info messages written to screen
seed          = 17
nDigits       = int(np.log10(nChainLength)) + 1 # Number of digits in appended number

# Pickle entire sample chain (cummulative, and therefore not exactly ideal, but simple)
def PickleResults(driver,filename):
    print("Writing output: "+filename)
    x = driver.sampler.chain
    OutNames = ['x',    # The names here must be the exact variable names
                'ndim',
                'ndata',
                'nwalkers',
                'nBurnIn',
                'nChainLength',
                'iters']

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
    result = driver.Eval(x)
    if result > 0:
        return -np.inf
    return result

# Build the persistent class containing the driver object
driver = DriverWrap()
driver.d = pymc.Driver(len(sys.argv), sys.argv, 1)
driver.d.SetComm(MPI.COMM_WORLD)
driver.d.init(len(sys.argv),sys.argv)

# Hang on to this for later - only do output on rank 0
rank = MPI.COMM_WORLD.Get_rank()

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

# Initialize the sampler with the chosen specs.
driver.sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[driver])
driver.sampler._random =  np.random.mtrand.RandomState(seed=seed) # overwrite state of rand with seeded one

# Generate initial samples
p0 = [prior_mean + driver.sampler._random.randn(ndim) * ensemble_std for i in xrange(nwalkers)]
        
print('Initial walker parameters: ')
for walker in p0:
    print(walker)

# Run burn-in steps
print ('Doing burn-in...')
pos, prob, state = driver.sampler.run_mcmc(p0, nBurnIn)

# Reset the chain to remove the burn-in samples.
driver.sampler.reset()

# Starting from the final position in the burn-in chain, do sample steps.
print ('Sampling...')
iters = 0
for result in driver.sampler.sample(pos, iterations=nChainLength):
    iters = iters + 1

    if rank == 0:
        if iters % runlogPeriod == 0:
            print(str(iters)+" walker sweeps complete")

        if iters % outFilePeriod == 0:
            fmt = "%0"+str(nDigits)+"d"
            outFileName = outFilePrefix + (fmt % iters)
            PickleResults(driver,outFileName)

if iters % outFilePeriod != 0 and rank == 0:
    fmt = "%0"+str(nDigits)+"d"
    outFileName = outFilePrefix + (fmt % iters)
    PickleResults(driver,outFileName)


# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 250-dimensional
# vector.
if rank == 0:
    print("Mean acceptance fraction:", np.mean(driver.sampler.acceptance_fraction))

    posterior_mean = []
    for i in range(ndim):
        posterior_mean.append(driver.sampler.flatchain[:,i].mean()) 
        print('New mean:',i,posterior_mean[i])

