#
# UQBox code (pronounced like "jukebox" without the "j")
#
# chief software engineer: Marcus Day
# helper elf programmers:  Jonathan Goodman (goodman@cims.nyu.edu)
#                          John Bell
#                          Matthias Morzfeld
# lost reindeer:           Ray Grout (ray.grout@nrel.gov)

from __future__ import print_function
print("Welcome to UqBox parallel. Be brave")
import numpy as np
import emcee
import sys
import pyemcee as pymc
import cPickle
import git_info as gi

# serial UqBox, Boxlib parallizes experiments (over MPI)
parallel_mode = 'BOXLIB'

# MPI parallel UqBox, boxlib threads experiments with OMP
parallel_mode = 'HYBRID'

print("Parallel mode is: " + parallel_mode)

if parallel_mode:
    from mpi4py import MPI


def WritePlotfile(driver, outFilePrefix, nwalkers,
                  step, nSteps, nDigits, rstate, id, rl):

    fmt = "%0"+str(nDigits)+"d"
    lastStep = step + nSteps - 1
    filename = outFilePrefix + '_' + (fmt % step) + '_' + (fmt % lastStep)

    if rank == 0:
        print('Writing plotfile: '+filename)

    x = driver.sampler.chain

    ndim = driver.NumParams()
    C_array_size = nSteps*ndim*nwalkers
    x_for_c = pymc.DoubleVec(C_array_size)

    for walker in range(0, nwalkers):
        for it in range(0, nSteps):
            for dim in range(0, ndim):
                index = walker + nwalkers*it + nwalkers*nSteps*dim
                x_for_c[index] = x[walker, it, dim]

    if rstate is None:
        rstateString = ''
    else:
        rstateString = cPickle.dumps(rstate)

    pf = pymc.UqPlotfile(x_for_c, ndim, nwalkers, step, nSteps, rstateString)
    pf.Write(filename)

    if rank == 0:
        flog=open('plotfiles.log', 'a')
        flog.write(id + "  " + rl + " " + filename)
        flog.close()

def LoadPlotfile(driver,filename):
    
    if rank == 0:
        print('Loading plotfile: '+filename)

    pf = pymc.UqPlotfile()
    pf.Read_serial(filename)
    if rank == 0:
        print('Done Loading plotfile: '+filename)
    t_nwalkers = pf.NWALKERS()
    t_ndim = pf.NDIM()
    t_iters = 1

    rstate = cPickle.loads(pf.RSTATE())
    iter = pf.ITER() + pf.NITERS() - 1

    p0 = pf.LoadEnsemble(iter, t_iters)

    ret = []
    for walker in range(0, t_nwalkers):
        ret.append(np.zeros(t_ndim))
        for dim in range(0, t_ndim):
            ret[walker][dim] = p0[walker + t_nwalkers*dim]

    return ret, iter, rstate


#
# Simple driver to enable persistent static class wrapped around driver object
#
# Construction of the contained "Driver" will read the input file listed on the
#  command line, set up the active parameters and synthetic experiments
#  described therein.
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
#  of sampled values of the parameters, and to deal with bad evals.  Here
#  a bad eval is signaled as a positive return value.  In this case we
#  set the result to -infinity, a special result recognized by emcee.
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

    f = open('hist_likelyhood_'+str(rank), 'a')
    for d in x:
        f.write(str(d))
        f.write(' ')

    f.write(str(result))
    f.write('\n')

    if result > 0:
        return -np.inf
    return result


def argfcn(x):
    return x

print('Setting up evaluator')
# Build the persistent class containing the driver object
driver = DriverWrap()
driver.d = pymc.Driver(len(sys.argv), sys.argv, 1)
driver.d.SetComm(MPI.COMM_WORLD)
driver.d.SetNumThreads(24)  # MUST be called before driver init
print('Calling evaluator init')
driver.d.init(len(sys.argv), sys.argv)

if parallel_mode == 'HYBRID':
    driver.d.SetParallelModeThreaded()


# Hang on to this for later - only do output on rank 0
rank = MPI.COMM_WORLD.Get_rank()

ndim = driver.NumParams()
ndata = driver.NumData()
prior_mean = driver.PriorMean()
prior_std = driver.PriorStd()
ensemble_std = driver.EnsembleStd()

pp = pymc.ParmParse()
print('Setting up sampler')

nwalkers = int(pp['nwalkers'])
maxStep = int(pp['maxStep'])
outFilePrefix = pp['outFilePrefix']
outFilePeriod = int(pp['outFilePeriod'])
seed = int(pp['seed'])
restartFile = pp['restartFile']
emcee_stepsize = float(pp['emcee_stepsize'])

if rank == 0:
    print('     nwalkers: ', nwalkers)
    print('      maxStep: ', maxStep)
    print('outFilePrefix: ', outFilePrefix)
    print('outFilePeriod: ', outFilePeriod)
    print('         seed: ', seed)
    print('  restartFile: ', restartFile)
    print('')

    print('Number of Parameters:', ndim)
    print('Number of Data:', ndata)
    print('prior means:  ' + str(prior_mean))
    print('prior std: ' + str(prior_std))
    print('ensemble std: ' + str(ensemble_std))
    print('emcee stepsize: ' + str(emcee_stepsize))


# Function to run sampler
def do_sampler():
    if restartFile == "":

        # Choose an initial set of positions for the walkers.
        # overwrite state of rand with seeded one
        driver.sampler._random = np.random.mtrand.RandomState(seed=seed)
        p0 = [prior_mean + driver.sampler._random.randn(ndim)
              * ensemble_std for i in xrange(nwalkers)]

        step = 0

        have_state = False

    else:
        if rank == 0:
            print ('Rstarting from '+restartFile)

        pos, step, state = LoadPlotfile(driver, restartFile)
        step = step + 1

        have_state = True

    # Main sampling loop
    if rank == 0:
        print ('Sampling...')

    if rank == 0:
        g = open('accept', 'w')

    pbsid = 0
    gitrl = 0
    if rank == 0:
        pbsid = gi.get_pbsid()
        gitrl = gi.get_reflog()
        patch = gi.get_patch()
        patchf = open(id + ".src", 'w')
        patchf.write(patch)
        patchf.close()
        
    while step < maxStep:
        nSteps = min(outFilePeriod, maxStep - step + 1)

        driver.sampler.reset()

        if have_state:
            pos, prob, state = driver.sampler.run_mcmc(pos, nSteps,
                                                       rstate0=state)
        else:
            pos, prob, state = driver.sampler.run_mcmc(p0, nSteps)
            have_state = True

        if rank == 0:
            print("Mean acceptance fraction:",
                  np.mean(driver.sampler.acceptance_fraction))
            g.write('Mean acceptance fraction:'
                    + str(np.mean(driver.sampler.acceptance_fraction)) + '\n')
            g.flush()

        nDigits = int(np.log10(maxStep)) + 1
        WritePlotfile(driver, outFilePrefix, nwalkers,
                      step, nSteps, nDigits, state, pbsid, gitrl)
    
        step = step + nSteps

# Build a sampler object

if parallel_mode == 'HYBRID':
    from UqBox_pool import UqBoxPool
    pool = UqBoxPool(MPI=MPI, debug=False)
    pool.set_function(lnprob)
    pool.set_function_arg(driver)
    if not pool.is_master():
        pool.wait()
    else:
        print('parallel mode: ' + parallel_mode)
        driver.sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                               argfcn, a=emcee_stepsize,
                                               pool=pool)
        do_sampler()
    pool.close()
    print("Done everything and pool closed up for rank " + str(rank))

else:
    driver.sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                           lnprob, args=[driver],
                                           a=emcee_stepsize)
