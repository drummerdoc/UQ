import numpy as np
import sys
import scipy.linalg
import pyemcee as pymc
from mpi4py import MPI

#
# Reading and writing (plot-)files
#
def WritePlotfile(samples,ndim,outFilePrefix,nwalkers,step,nSteps,nDigits,rstate):
    
    if nDigits > 0:
        fmt = "%0"+str(nDigits)+"d"
        lastStep = step + nSteps - 1
        filename = outFilePrefix + '_' + (fmt % step) + '_' + (fmt % lastStep)
    else:
        filename = outFilePrefix

    if rank == 0:

        print('Writing plotfile: '+filename)
        
        C_array_size = nSteps*ndim*nwalkers
        x_for_c = pymc.DoubleVec(C_array_size)

        for walker in range(0,nwalkers):
            for it in range(0,nSteps):
                for dim in range(0,ndim):
                    index = walker + nwalkers*it + nwalkers*nSteps*dim
                    x_for_c[index] = samples[dim,step+it]

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
    return -result

# Build the persistent class containing the driver object
driver = DriverWrap()
driver.d = pymc.Driver(len(sys.argv), sys.argv, 1)
#driver.d.SetComm(MPI.COMM_WORLD)
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
infl              = float(pp['infl'])
nu                = float(pp['nu'])
use_plt           = int(pp['use_plt'])

if rank == 0:
    print '      maxStep: ',maxStep
    print 'outFilePrefix: ',outFilePrefix
    print 'outFilePeriod: ',outFilePeriod
    print '         seed: ',seed
    print '  restartFile: ',restartFile
    print '         neff: ',neff
    print 'inflation factor:', infl
    if whichSampler == 1:
        print 'Gaussian sampling'
    else:
        print 'multivariate t'
        print 'nu = ',nu
    print ''

    print 'Number of Parameters:',ndim
    print 'Number of Data:',ndata
    print 'prior means:  '+ str(prior_mean)
    print 'prior std: '+ str(prior_std)
    print 'ensemble std: '+ str(ensemble_std)
    print ' '

NOS = maxStep
np.set_printoptions(linewidth=200)

def F0(x,mu,L2):
    y =  L2*(x-mu)
    return 0.5*(np.linalg.norm(y)**2)

def F0t(x,mu,L2):
    y =  L2*(x-mu)
    return (np.linalg.norm(y)**2)

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
    N = samples.shape[0]
    M = samples.shape[1]
    c = np.zeros(M+1)
    for j in range(1,len(c)):
        c[j] = c[j-1]+w[j-1]
    i = 0
    #u1 = np.random.rand(1)/M
    u1 = np.random.uniform(0,1/M,1)
    u = 0
    rs_map = np.zeros(M, dtype=int)
    for m in range(M):
        u = u1 + float(m)/M
	while u >= c[i]:
            i += 1
        rs_map[m] = i-1 # Note: i is never 0 here
    return rs_map

# Evals and evecs of Covariance
def ComputeHessEvals(Cov,neff):
    #print 'Covariance:', Cov
    evals,evecs = np.linalg.eigh(Cov)
    evecs = np.matrix(evecs)
    
    #sl = np.argsort(evals)
    #evals = evals[sl]
    #evecs = evecs[:,sl]
    #evals = evals[-neff:]
    #evecs = evecs[:,-neff:]        

    L = evecs * np.diag(np.sqrt(evals))
    L2 = np.diag(np.sqrt(1/evals)) * evecs.T
    
    return evecs, evals, L, L2

# linear map                                                                                             
def LinearMap(NOS,N,neff,mu,Cov,lower_bounds,upper_bounds):
    print 'Sampling with linear maps'
    Samples = np.matrix(np.zeros(shape=(N,NOS)))
    Fo = np.matrix(np.zeros(shape=(NOS,1)))
    evecs,evals,L,L2 = ComputeHessEvals(Cov,neff)
    for i in range(NOS):
        sample_oob = True
        while sample_oob == True:
            Samples[:,i] = mu.T +  L*np.matrix(np.random.randn(neff,1))
            sample_good = True
            for n in range(N):
                sample_good &= Samples[n,i]>=lower_bounds[n] and Samples[n,i]<=upper_bounds[n]
            sample_oob = not sample_good

        Fo[i] = F0(Samples[:,i],mu.T,L2)

    return Samples, Fo


def tDist(nu,NOS,N,neff,mu,Cov,lower_bounds,upper_bounds):
    print 'Sampling with t distribution'
    Samples = np.matrix(np.zeros(shape=(N,NOS)))
    Fo = np.matrix(np.zeros(shape=(NOS,1)))
    evecs,evals,L,L2 = ComputeHessEvals(Cov,neff)
    for i in range(NOS):
        sample_oob = True
        while sample_oob == True:
            u = np.random.chisquare(nu)
            Samples[:,i] = mu.T + (np.sqrt(nu/u))*(L*np.matrix(np.random.randn(neff,1)))
            sample_good = True
            for n in range(N):
                sample_good &= Samples[n,i]>=lower_bounds[n] and Samples[n,i]<=upper_bounds[n]
            sample_oob = not sample_good

        Fo[i] = ((nu+neff)/2) * np.log( 1 + F0t(Samples[:,i],mu.T,L2)/nu )
    return Samples, Fo

#
# MAIN PROGRAM
#
np.random.seed(seed=seed)

if use_plt == 1:
    print 'Using plotfiles to read data'
    Samples1 = LoadPlotfile(initialSamples)#'GMM_It5_RS')
    N = Samples1[0].shape[0]
    M = min(len(Samples1),numInitialSamples)
    Samples2 = Samples1[:M]
    print "using ",len(Samples2),"samples"
    data = np.array(np.zeros(shape=(M,Samples2[0].shape[0]+1)))
    Samples2.reverse() # reversed to agree with original data
    for i,v in enumerate(Samples2):
        data[i,:-1] = v
else:
    data = np.loadtxt(initialSamples)
    N = data.shape[-1] - 1          # Number of independent variables
    M = min(data.shape[0],numInitialSamples)  # Max number of data points to use
    data = data[:M,:]
    print 'Using ',M,' samples to construct local model'


scales = prior_mean             # Scale values for independent data
lower_bounds = np.array(driver.LowerBound())/scales
upper_bounds = np.array(driver.UpperBound())/scales

print 'lower bounds'
print lower_bounds
print 'upper bounds'
print upper_bounds

x = data[-M:,:N].copy() 
scaled_x = np.copy(x)
for i in range(M):
    scaled_x[i] = x[i]/scales
mu = np.matrix(np.mean(scaled_x,axis=0))
print 'mean = ', mu
print 'infl =',infl
#print type(infl)
Cov = infl*np.cov(scaled_x.T)

if whichSampler == 1: # use Gaussian
    if rank == 0:
         Samples1, Fo = LinearMap(NOS,N,neff,mu,Cov,lower_bounds,upper_bounds)    
elif whichSampler == 2: # use t-distribution
    if rank == 0:
         Samples1, Fo = tDist(nu,NOS,N,neff,mu,Cov,lower_bounds,upper_bounds)
else: 
    print "What sampler would you like to use?"

print 'Building Samples matrix'
Samples = np.matrix(np.zeros(shape=(N,NOS)))
print 'done building Samples matrix'
#for i in range(NOS):
#    if rank == 0:
#        Samples[:,i] =  np.multiply(Samples1[:,i].T,np.matrix(scales)).T

Samples1 = 1 # Clear this array?

#print np.shape(Samples)
if rank == 0:
    nwalkers = 1
    nDigits = 0
    istart = 0
    chunkSize = 500000
    while istart < NOS:
        inum = min(NOS-istart,chunkSize)
        numStr = ('%08d_%08d') % (istart,istart+inum)
        filename =  outFilePrefix + "_initSamples_" + numStr
        WritePlotfile(Samples,N,filename,nwalkers,istart,inum,nDigits,None)
        WritePlotfile(Fo.T,1,filename + "/F0",nwalkers,istart,inum,nDigits,None)
        istart += inum
