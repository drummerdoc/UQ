import numpy as np
import sys
import scipy.linalg


import pyemcee as pymc
import cPickle
from mpi4py import MPI

def WritePlotfile(samples,ndim,outFilePrefix,nwalkers,step,nSteps,nDigits,rstate):
    
    fmt = "%0"+str(nDigits)+"d"
    lastStep = step + nSteps - 1
    filename = outFilePrefix + '_' + (fmt % step) + '_' + (fmt % lastStep)

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
    t_nwalkers = pf.NWALKERS()
    t_ndim = pf.NDIM()
    t_iters = 1

    rstate = cPickle.loads(pf.RSTATE())
    iter = pf.ITER() + pf.NITERS() - 1

    p0 = pf.LoadEnsemble(iter,t_iters)

    ret = []
    for walker in range(0,t_nwalkers):
        ret.append(np.zeros(t_ndim))
        for dim in range(0,t_ndim):
            ret[walker][dim] = p0[walker + t_nwalkers*dim]

    return ret, iter, rstate


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
neff2             = int(pp['neff2'])

if rank == 0:
    print '      maxStep: ',maxStep
    print 'outFilePrefix: ',outFilePrefix
    print 'outFilePeriod: ',outFilePeriod
    print '         seed: ',seed
    print '  restartFile: ',restartFile
    print '         neff: ',neff
    print '        neff2: ',neff2
    print ''

    print 'Number of Parameters:',ndim
    print 'Number of Data:',ndata
    print 'prior means:  '+ str(prior_mean)
    print 'prior std: '+ str(prior_std)
    print 'ensemble std: '+ str(ensemble_std)

    
NOS = maxStep
np.set_printoptions(linewidth=200)

def Fo(x,phi,mu,evecs,evals):
    y =  np.multiply(1/np.sqrt(evals).T,evecs.T*(x-mu))
    return phi + 0.5*(np.linalg.norm(y)**2)

def Fo2(x,phi,mu,L):
    y = np.linalg.solve(L,(x-mu))
    return phi + 0.5*(np.linalg.norm(y)**2)

def EffSampleSize(w):
    n = len(w)
    sumSq = 0
    for i in range(n):
        sumSq += w[i]*w[i]
    if sumSq ==0:
        return 0
    return 1/sumSq

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

def WeightedMean(w, samples):
    N = samples.shape[0]
    M = samples.shape[1]
    CondMean = np.zeros(N)
    for n in range(N):
        for m in range(M):
            CondMean[n] += w[m]*samples[n,m]
    return CondMean

def WeightedVar(CondMean, w, samples):
    N = samples.shape[0]
    M = samples.shape[1]
    CondVar = np.zeros(N)
    for n in range(N):
        for m in range(M):
            CondVar[n] += w[m]*(samples[n,m]-CondMean[n])*(samples[n,m]-CondMean[n])
    return CondVar

def Resampling(w,samples):
    N = samples.shape[0]
    M = samples.shape[1]
    c = np.zeros(M+1)
    for j in range(1,len(c)):
        c[j] = c[j-1]+w[j-1]
    i = 0
    u1 = np.random.rand(1)/M
    u = 0
    rs_map = np.zeros(M, dtype=int)
    for m in range(M):
        u = u1 + float(m)/M
        while u >= c[i]:
            i += 1
        rs_map[m] = i-1 # Note: i is never 0 here
    return rs_map

# Read data
data = np.loadtxt(initialSamples)
N = data.shape[-1] - 1          # Number of independent variables
M = numInitialSamples           # Max number of data points to use
scales = prior_mean             # Scale values for independent data

x = data[-M:,:N]                # Independent data
z = np.matrix(data[-M:, -1]).T  # Dependent data, as column vector

scaled_x = np.copy(x)

## Matti's local model
for i in range(M):
    scaled_x[i] = x[i]/scales

Hinv = np.cov(scaled_x.T)       # NB: Possibly scale by "inflation factor" to be optimized

from numpy import linalg as LA
#evals,evecs = LA.eig(Hinv)
evals,evecs = LA.eigh(Hinv)
evecs = np.matrix(evecs)
sl = np.argsort(evals)
evals = evals[sl]
evecs = evecs[:,sl]

if rank == 0:
    print 'Eigenvalues:',evals
evals = np.matrix(evals[-neff:])
evecs = evecs[:,-neff:]
if rank == 0:
    print 'Eigenvalues kept:',evals

mu = np.matrix(x[-1,:]/scales).T
phi = -data[-1,-1]              # Select phi where F is minimum in data set

# Compute residual
# residual = []
# for i in range(M):
#     Fzero = Fo(np.matrix(scaled_x[i]).T,phi,mu,evecs,evals)
#     residual.append(Fzero + z[[i]])

Samples = np.matrix(np.zeros(shape=(N,NOS)))
w = np.array(np.zeros(NOS))
newF = np.array(np.zeros(NOS))
F0 = np.array(np.zeros(NOS))

lower_bounds = np.array(driver.LowerBound())/scales
upper_bounds = np.array(driver.UpperBound())/scales

for i in range(NOS):

    sample_oob = True
    while sample_oob == True:
        Samples[:,i] = mu + evecs*np.multiply(np.sqrt(evals), np.random.randn(1,neff)).T
        sample_good = True
        for n in range(N):
            sample_good &= Samples[n,i]>=lower_bounds[n] and Samples[n,i]<=upper_bounds[n]
        sample_oob = not sample_good
    
    F0[i] = Fo(Samples[:,i],phi,mu,evecs,evals)
    Samples[:,i] =  np.multiply(Samples[:,i].T,np.matrix(scales)).T
    xx = np.array(Samples[:,i].T)[0]
    newF[i] = -lnprob(xx,driver)
    if newF[i] == np.inf:
        w[i] = -1
    else:
        w[i] = F0[i] - newF[i]

    if rank == 0:
        print "Sample ",i,"of",NOS,"F0 =",F0[i]," F =",newF[i]
        
if rank == 0:
    wmax = np.amax(w)
    for i in range(NOS):
        if w[i] < 0:
            w[i] = 0
        else:
            w[i] = np.exp(w[i] - wmax)

    wsum = np.sum(w)
    w = w/wsum

    print 'Effective sample size: ',EffSampleSize(w)
    print 'Quality measure R:',CompR(w)

    CondMean = WeightedMean(w,Samples)
    print 'Conditional mean: ',CondMean
    print 'Conditional std: ',np.sqrt(WeightedVar(CondMean,w,Samples))

    rs_map = Resampling(w,Samples)
    Xrs = Samples[:,rs_map]

    # nwalkers = 1
    # if NOS > 0:
    #     nDigits = int(np.log10(NOS)) + 1
    #     WritePlotfile(Samples,N,outFilePrefix,nwalkers,0,NOS,nDigits,None)
    #     WritePlotfile(Xrs,N,outFilePrefix+'_RS',nwalkers,0,NOS,nDigits,None)

    CondMeanRs = WeightedMean(np.ones(NOS)/NOS,Xrs)
    print 'Conditional mean after resampling: ',CondMeanRs
    print 'Conditional std after resampling: ',np.sqrt(WeightedVar(CondMeanRs,np.ones(NOS)/NOS,Xrs))
else:
    rs_map = np.arange(0, NOS, dtype=int)

comm.Barrier()
M = NOS
    
if rank == 0:
    x = Samples[:, rs_map].T.copy()
    z = newF[rs_map].copy()

    scaled_x = np.copy(x)
    for i in range(M):
        scaled_x[i] = x[i]/scales
        
    print 'x',x
    print 'scaled_x',scaled_x
    
    # Matti's local model
    Hinv = np.cov(scaled_x.T)       # NB: Possibly scale by "inflation factor" to be optimized

    print 'Hinv',Hinv

    #evals,evecs = LA.eig(Hinv)
    evals,evecs = LA.eigh(Hinv)
    print 'evals',evals
    
    evecs = np.matrix(evecs)
    sl = np.argsort(evals)
    evals = evals[sl]
    evecs = evecs[:,sl]

    print 'Eigenvalues:',evals
    evals = np.matrix(evals[-neff2:])
    evecs = evecs[:,-neff2:]
    print 'Eigenvalues kept:',evals

    
for i in range(NOS):

    if rank == 0:

        sample_oob = True
        while sample_oob == True:
            Samples[:,i] = mu + evecs*np.multiply(np.sqrt(evals), np.random.randn(1,neff2)).T
            sample_good = True
            for n in range(N):
                sample_good &= Samples[n,i]>=lower_bounds[n] and Samples[n,i]<=upper_bounds[n]
            sample_oob = not sample_good

    F0[i] = Fo(Samples[:,i],phi,mu,evecs,evals)
    Samples[:,i] =  np.multiply(Samples[:,i].T,np.matrix(scales)).T
    xx = np.array(Samples[:,i].T)[0]
    newF[i] = -lnprob(xx,driver)
    if newF[i] == np.inf:
        w[i] = -1
    else:
        w[i] = F0[i] - newF[i]

    if rank == 0:
        print "Sample ",i,"of",NOS,"F0 =",F0[i]," F =",newF[i]


if rank == 0:
    wmax = np.amax(w)
    for i in range(NOS):
        if w[i] < 0:
            w[i] = 0
        else:
            w[i] = np.exp(w[i] - wmax)

    wsum = np.sum(w)
    w = w/wsum

    print 'Effective sample size: ',EffSampleSize(w)
    print 'Quality measure R:',CompR(w)
    
    CondMean = WeightedMean(w,Samples)
    print 'Conditional mean: ',CondMean
    print 'Conditional std: ',np.sqrt(WeightedVar(CondMean,w,Samples))

    rs_map = Resampling(w,Samples)
    Xrs = Samples[:,rs_map]

    # nwalkers = 1
    # if NOS > 0:
    #     nDigits = int(np.log10(NOS)) + 1
    #     WritePlotfile(Samples,N,outFilePrefix,nwalkers,0,NOS,nDigits,None)
    #     WritePlotfile(Xrs,N,outFilePrefix+'_RS',nwalkers,0,NOS,nDigits,None)

    CondMeanRs = WeightedMean(np.ones(NOS)/NOS,Xrs)
    print 'Conditional mean after resampling: ',CondMeanRs
    print 'Conditional std after resampling: ',np.sqrt(WeightedVar(CondMeanRs,np.ones(NOS)/NOS,Xrs))
else:
    rs_map = np.zeros(M, dtype=int)

