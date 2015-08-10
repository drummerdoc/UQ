import numpy as np
import sys
import scipy.linalg

import pickle
import pyemcee as pymc
from mpi4py import MPI

def genName(nDigits, outFilePrefix, step, nSteps):
    if nDigits > 0:
        fmt = "%0"+str(nDigits)+"d"
        lastStep = step + nSteps - 1
        filename = outFilePrefix + '_' + (fmt % step) + '_' + (fmt % lastStep)
    else:
        filename = outFilePrefix
    return filename


def WritePlotfile(samples,ndim,filename,nwalkers,step,nSteps,nDigits,rstate):
    if rank == 0:
        print('Writing plotfile: '+filename)
        
        C_array_size = nSteps*ndim*nwalkers
        x_for_c = pymc.DoubleVec(C_array_size)

        for walker in range(0,nwalkers):
            for it in range(0,nSteps):
                for dim in range(0,ndim):
                    index = walker + nwalkers*it + nwalkers*nSteps*dim
                    x_for_c[index] = samples[dim,it]

        if rstate == None:
            rstateString = ''
        else:
            rstateString = cPickle.dumps(rstate)

        pf = pymc.UqPlotfile(x_for_c, ndim, nwalkers, step, nSteps, rstateString)
        pf.Write(filename)


def LoadPlotfile(filename):
    
    if rank == 0:
        print('Loading plotfile: '+filename)
        
    pf = pymc.UqPlotfile()
    pf.Read(filename)
    nwalkers = pf.NWALKERS()
    ndim = pf.NDIM()
    iters = pf.NITERS()
    iter = pf.ITER()

    p0 = pf.LoadEnsemble(iter,iters)

    ret = []
    for walker in range(0,nwalkers):
        for it in range(0,iters):
            ret.append(np.zeros(ndim))
            for dim in range(0,ndim):
                index = walker + nwalkers*it + nwalkers*iters*dim
                ret[-1][dim] = p0[index]

    return ret


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
    def LowerBound(self):
        return self.d.LowerBound()
    def UpperBound(self):
        return self.d.UpperBound()
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
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

ndim = driver.NumParams()
ndata = driver.NumData()
prior_mean = driver.PriorMean()
prior_std = driver.PriorStd()
ensemble_std = driver.EnsembleStd()

pp = pymc.ParmParse()

maxStep           = int(pp['maxStep'])
outFilePrefix     =     pp['outFilePrefix']
outFilePeriod     = int(pp['outFilePeriod'])
seed              = int(pp['seed'])
restartFile       =     pp['restartFile']
initialSamples    =     pp['initial_samples']
numInitialSamples = int(pp['num_initial_samples'])
neff              = int(pp['neff'])
whichSampler      = int(pp['whichSampler'])

if rank == 0:
    print '      maxStep: ',maxStep
    print 'outFilePrefix: ',outFilePrefix
    print 'outFilePeriod: ',outFilePeriod
    print '         seed: ',seed
    print '  restartFile: ',restartFile
    print '         neff: ',neff
    print ''

    print 'Number of Parameters:',ndim
    print 'Number of Data:',ndata
    print 'prior means:  '+ str(prior_mean)
    print 'prior std: '+ str(prior_std)
    print 'ensemble std: '+ str(ensemble_std)

    
NOS = maxStep
np.set_printoptions(linewidth=200)


def EffSampleSize(w):
    n = w.shape[0]
    R = CompR(w)
    return n/R

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

def CompRN(w): # normalized weights only
    N = len(w)
    w2 = 0
    for i in range(N):
        w2 += w[i]*w[i]
    return N*w2

def Resampling(w,samples):
    M = samples.shape[0]
    N = samples.shape[1]
    c = np.zeros(M+1)
    for j in range(1,len(c)):
        c[j] = c[j-1]+w[j-1]
    i = 0
    #u1 = np.random.rand(1)/M
    u1 = np.random.uniform(0,1.0/M,1)
    u = 0
    rs_map = np.zeros(M, dtype=int)
    for m in range(M):
        u = u1 + float(m)/M
        while u >= c[i]:
            i += 1
        rs_map[m] = i-1 # Note: i is never 0 here
    return rs_map

def whist(x,w,nbins):
    xl = np.min(x)
    xr = np.max(x)
    dx = (xr - xl)/nbins
    bins = np.zeros(nbins)
    for i,xx in enumerate(x):
        index = int(np.floor((xx - xl)/dx))
        if index >= 0 | index < nbins:
            bins[index] += w[i]
    bins *= 1/dx
    ax = np.arange(xl+dx/2,xr+dx/2,dx)
    return ax,bins

########################################################################################################################################
# MAIN PROGRAM
########################################################################################################################################

if rank == 0:
    print 'Starting to re-weight with full model'

filename =  outFilePrefix + "_initSamples"
Fo_file = LoadPlotfile(filename + "/F0")
F_file = LoadPlotfile(filename + '/Ffile')

F = []
Fo = []
for i in range(len(F_file)):
    F.append(-F_file[i][0])
    Fo.append(Fo_file[i][0])

Samples = LoadPlotfile(filename)

NOS = len(Samples)

w = np.array(np.zeros(shape=(NOS,1)))
for i in range(NOS):
    if rank == 0:
        if F[i] <= 0:
            w[i] = np.inf
        else:
            w[i] = Fo[i] - F[i]
                
if rank == 0:

    good_inds = np.nonzero(np.isinf(w)==0)[0]
    good_NOS = np.shape(good_inds)[0]
    w = w[good_inds]
    F = np.matrix(F).T
    F = F[good_inds,:]
    
    Samples = np.matrix(Samples).T
    Samples = Samples[:,good_inds]
    wmax = np.amax(w)
    
    for i in range(good_NOS):
        if w[i] == np.inf:
            w[i] = 0
        else:
            w[i] = np.exp(w[i] - wmax)
    
    wsum = np.sum(w)
    w = w/wsum
    
    Samples = np.matrix(Samples).T
    
    print 'Effective sample size: ',EffSampleSize(w)
    R = CompRN(w)
    print 'Quality measure R:',R

    rs_map = Resampling(w,Samples)
    Xrs = Samples[rs_map,:].T
    Frs = F[rs_map,:].T

    nwalkers = 1
    if good_NOS > 0:
         nDigits = int(np.log10(good_NOS)) + 1
         M = Xrs.shape[0]
         filename = genName(nDigits, outFilePrefix+'_RS', 0, good_NOS)
         WritePlotfile(Xrs,M,filename,nwalkers,0,good_NOS,nDigits,None)
         WritePlotfile(Frs,1,filename+'/F',1,0,good_NOS,0,None)

         fmt = "%0"+str(nDigits)+"d"
         lastStep = good_NOS - 1
         filename = outFilePrefix + '_RS_' + (fmt % 0) + '_' + (fmt % (good_NOS-1))
         pickle.dump(w,open(filename+"/w.pic", "wb" ) )
         pickle.dump(R,open(filename+"/R.pic", "wb" ) )
