import numpy as np
import sys
import scipy.linalg

import scipy.optimize as optimize
import scipy.interpolate as inter
import dill
import klepto
import cPickle as pickle

import pyemcee as pymc
import cPickle
from mpi4py import MPI

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
                    if ndim==1:
                        x_for_c[index] = samples[it]
                    else:
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
ampFactor         = int(pp['ampFactor'])
stage             = int(pp['stage'])
buildRBF          = int(pp['buildRBF'])
rbfLoadFile       =     pp['rbfLoadFile']
rbfDumpFile       =     pp['rbfDumpFile']
minimizeRBF       = int(pp['minimizeRBF'])
optLoadFile       =     pp['optLoadFile']
optDumpFile       =     pp['optDumpFile']
whichSampler      = int(pp['whichSampler'])

if rank == 0:
    print '      maxStep: ',maxStep
    print 'outFilePrefix: ',outFilePrefix
    print 'outFilePeriod: ',outFilePeriod
    print '         seed: ',seed
    print '  restartFile: ',restartFile
    print '         neff: ',neff
    print '    ampFactor: ',ampFactor
    print ''

    print 'Number of Parameters:',ndim
    print 'Number of Data:',ndata
    print 'prior means:  '+ str(prior_mean)
    print 'prior std: '+ str(prior_std)
    print 'ensemble std: '+ str(ensemble_std)

    
NOS = maxStep
np.set_printoptions(linewidth=200)

def F0(x,phi,mu,L2):
    y =  L2*(x-mu)
    return phi + 0.5*(np.linalg.norm(y)**2)

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

def EvalRBF(x,rbf):
	tmp = rbf(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8])
	return tmp

def G(l,phi,mu,Leta,rho,rbf):
    x = (mu+l*Leta)
    F = EvalRBF(np.array(x)[0],rbf)
    return F - phi -0.5*rho


# Forward finite differences
def ffd(x,rbf,k,h):
    h = x[k] * h

    xpdx = x.copy()
    xpdx[k] = xpdx[k]+ h
    
    fx = EvalRBF(x,rbf)
    fxpdx = EvalRBF(xpdx,rbf)
    
    return (fxpdx - fx)/(xpdx[k]-x[k])


# Gradient of rbf approximate likelihood
def GradRBF(x,rbf,h):
    grad = np.matrix(np.zeros(shape=(N,1)))
    for i in range(N):
        grad[i] = ffd(x,rbf,i,h)
    return grad

# optimization with rbf                                                                       
def OptimizeRBF(rbf,lower_bounds,upper_bounds,scaled_x,scales):
    print 'Optimizing rbf model'
    bnds = []
    for i in range(N):
        bnds.append((lower_bounds[i],upper_bounds[i]))
    bnds = tuple(bnds)
    xopt = optimize.minimize(EvalRBF,scaled_x[-1,:],args = (rbf,),method='TNC',bounds=bnds,options=dict({'maxiter':1000}))
    print 'Minimizer of rbf:', np.multiply(np.matrix(xopt.x),np.matrix(scales))
    print 'Minimizer in hammer:', np.multiply(np.matrix(scaled_x[-1,:]),np.matrix(scales))
    print 'Minimum of rbf', xopt
    return xopt

# Evals and evecs of Hessian
def ComputeHessEvals(scaled_x,neff):
    Hinv = np.cov(scaled_x.T)
    evals,evecs = np.linalg.eigh(Hinv)
    evecs = np.matrix(evecs)
    sl = np.argsort(evals)
    evals = evals[sl]
    evecs = evecs[:,sl]
    
    evals = evals[-neff:]
    evecs = evecs[:,-neff:]        

    L = evecs * np.diag(evals)
    L2 = np.diag(np.sqrt(1/evals)) * evecs.T

    return evecs, evals, L, L2

# random map with rbf 
def RandomMap_rbf(NOS,N,neff,scaled_x,xopt,rbf,lower_bounds,upper_bounds):
    print 'Sampling with random maps and rbf'

    Samples = np.matrix(np.zeros(shape=(N,NOS)))
    Fo = np.matrix(np.zeros(shape=(NOS,1)))
    w  = np.array(np.zeros(shape=(NOS1,1)))

    evecs,evals,L,L2 = ComputeHessEvals(scaled_x,neff)
    phi = xopt.fun
    mu  = np.matrix(xopt.x)

    for i in range(NOS):
        sample_oob = True
        while sample_oob == True:
           xi  = L*np.matrix(np.random.randn(neff,1))
           rho = np.linalg.norm(xi)**2
           lopt1 = optimize.fsolve(G,1,args = (phi,mu,xi.T,rho,rbf,))#,epsfcn = 1e-5,xtol = 1e-6)
           lopt2 = optimize.fsolve(G,-1,args = (phi,mu,xi.T,rho,rbf,))
           if np.abs(1-lopt1)<np.abs(1-lopt2):
               lopt = lopt1
           else:
               lopt = lopt2

           print 'lopt1 = ',lopt1,'lopt2 = ',lopt2
           Samples[:,i] = (mu+lopt*xi.T).T
           sample_good = True
           for n in range(N):
               sample_good &= Samples[n,i]>=lower_bounds[n] and Samples[n,i]<=upper_bounds[n]
           sample_oob = not sample_good
        
        Fo[i] = EvalRBF(np.array(Samples[:,i].T)[0],rbf)
        print "Sample ",i , "of ", NOS, ', Fo = ',Fo[i],' lopt = ',lopt                                                   
                  
        gradF = GradRBF(Samples[:,i],rbf,1e-6)
        w[i] =  (neff-1)*np.log(np.abs(lopt)) + np.log(rho) - np.log(np.abs( xi.T*gradF ))                   

    # normalize weights                                                                                   
    wmax = np.amax(w)
    w = np.exp(w - wmax)
    wsum = np.sum(w)
    w = w/wsum

    return Samples, w, Fo


# linear map with RBF
def LinearMap_rbf(NOS,N,neff,scaled_x,xopt,rbf,lower_bounds,upper_bounds):
    print 'Sampling with linar maps and rbf'
    Samples = np.matrix(np.zeros(shape=(N,NOS)))
    Fo = np.matrix(np.zeros(shape=(NOS,1)))
    F_rbf = np.matrix(np.zeros(shape=(NOS,1)))
    w  = np.array(np.zeros(shape=(NOS1,1)))

    evecs,evals,L,L2 = ComputeHessEvals(scaled_x,neff)
    phi = xopt.fun
    mu  = np.matrix(xopt.x)
    
    for i in range(NOS):
        sample_oob = True
        while sample_oob == True:
            Samples[:,i] = mu.T +  L*np.random.randn(neff,1)
            sample_good = True
            for n in range(N):
                sample_good &= Samples[n,i]>=lower_bounds[n] and Samples[n,i]<=upper_bounds[n]
            sample_oob = not sample_good

        Fo[i] = F0(Samples[:,i],phi,mu.T,L2)
        #F_rbf[i] = EvalRBF(np.array(mu.T),rbf)

        F_rbf[i] = EvalRBF(np.array(Samples[:,i].T)[0],rbf)
        w[i] = Fo[i] - F_rbf[i]   
        print "Sample ", i+1, " of ", NOS, ", Fo = ",Fo[i], " F_rbf = ",F_rbf[i]

    # normalize weights   
    wmax = np.amax(w)
    w = np.exp(w - wmax)
    wsum = np.sum(w)
    w = w/wsum
    
    return Samples, w, Fo

# linear map                                                                                             
def LinearMap(NOS,N,neff,scaled_x,mu,phi,lower_bounds,upper_bounds):
    print 'Sampling with linar maps'
    Samples = np.matrix(np.zeros(shape=(N,NOS)))
    Fo = np.matrix(np.zeros(shape=(NOS,1)))

    evecs,evals,L,L2 = ComputeHessEvals(scaled_x,neff)
   
    for i in range(NOS):
        sample_oob = True
        while sample_oob == True:
            Samples[:,i] = mu.T +  L*np.random.randn(neff,1)
            sample_good = True
            for n in range(N):
                sample_good &= Samples[n,i]>=lower_bounds[n] and Samples[n,i]<=upper_bounds[n]
            sample_oob = not sample_good

        Fo[i] = F0(Samples[:,i],phi,mu.T,L2)
        print "Sample ", i+1, " of ", NOS, ", Fo = ",Fo[i]

    return Samples, Fo

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
            if stage == 2:
                w[i] = Fo[p] - F[i]
            else:
                w[i] = Fo[i] - F[i]
                
import matplotlib.pyplot as plt

if rank == 0:

    good_inds = np.nonzero(np.isinf(w)==0)[0]
    good_NOS = np.shape(good_inds)[0]
    w = w[good_inds]

    plt.plot(w)
    plt.show()
    
    Samples = np.matrix(Samples).T
    Samples = Samples[:,good_inds]
    wmax = np.amax(w)
    
    for i in range(good_NOS):
        if w[i] == np.inf:
            w[i] = 0
        else:
            w[i] = np.exp(w[i] - wmax)

    plt.plot(w)
    plt.show()
    
    wsum = np.sum(w)
    w = w/wsum

    #plt.plot(w*good_NOS)
    #plt.show()

    #nbins = 100
    #axw,binsw = whist(w,np.ones(good_NOS)/float(good_NOS),nbins)
    #plt.plot(axw,binsw,'b')
    #plt.show()
    
    Samples = np.matrix(Samples).T
    N = Samples.shape[0]
    
    print 'Effective sample size: ',EffSampleSize(w)
    R = CompRN(w)
    print 'Quality measure R:',R


