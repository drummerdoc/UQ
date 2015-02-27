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

def G(l,phi,mu,eta,rho,rbf):
	x = mu+l*eta
	F = EvalRBF(np.array(x)[0],rbf)
#	if rank == 0:
#		print 'G =',F - phi -0.5*rho
	return F - phi -0.5*rho

# Read data
data = np.loadtxt(initialSamples)
N = data.shape[-1] - 1          # Number of independent variables
M = numInitialSamples           # Max number of data points to use
scales = prior_mean             # Scale values for independent data

if rank ==0:
	x = data[-M:,:N].copy()         # Independent data
	z = -np.matrix(data[-M:, -1]).T.copy() # Dependent data, as column vector

	for i in range(M):
		x[M-i-1] = data[(data.shape[0]-i*10-1),:N]
		z[M-i-1] = -data[(data.shape[0]-i*10-1), -1]

ampFactor = 100;
NOS1 = ampFactor * NOS
w1 = np.array(np.zeros(shape=(NOS1,1)))
Samples1 = np.matrix(np.zeros(shape=(N,NOS1)))
Fo = np.matrix(np.zeros(shape=(NOS1,1)))
NEffSamples = 0
Xrs = np.matrix(np.zeros(shape=(N,NOS1)))

if rank == 0:
	scaled_x = np.copy(x)
	for i in range(M):
		scaled_x[i] = x[i]/scales

	lower_bounds = np.array(driver.LowerBound())/scales
	upper_bounds = np.array(driver.UpperBound())/scales

	# radial basis function model
	from scipy.interpolate import Rbf
	print 'Building rbf model' 
	rbf = Rbf(scaled_x[:,0],scaled_x[:,1],scaled_x[:,2],scaled_x[:,3],scaled_x[:,4],scaled_x[:,5],scaled_x[:,6],scaled_x[:,7],scaled_x[:,8],z)#,function = 'cubic')
	
	# optimization
        print 'Optimizing with rbf model'
	import scipy.optimize as optimize
	bnds = []
	for i in range(N):
		bnds.append((lower_bounds[i],upper_bounds[i]))
	bnds = tuple(bnds)
	xopt = optimize.minimize(EvalRBF,scaled_x[-1,:],args = (rbf,),method='TNC',bounds=bnds,options=dict({'maxiter':1000}))
	print 'Minimizer of rbf:', np.multiply(np.matrix(xopt.x),np.matrix(scales)) 
	print 'Minimizer in hammer:', np.multiply(np.matrix(scaled_x[-1,:]),np.matrix(scales))
	print 'Minimum of rbf', xopt

	# sampling
	print 'Sampling'
	if neff > 0:
		Hinv = np.cov(scaled_x.T) 
		evals,evecs = np.linalg.eigh(Hinv)
		evecs = np.matrix(evecs)
		sl = np.argsort(evals)
		evals = evals[sl]
		evecs = evecs[:,sl]
		evals = np.matrix(evals[-neff:])
		evecs = evecs[:,-neff:]
	
	phi = xopt.fun
	mu  = np.matrix(xopt.x)
	
	for i in range(NOS1):
		sample_oob = True
    		while sample_oob == True:
        		if neff > 0:
				xi  = np.matrix(np.random.randn(1,neff))
        			rho = np.linalg.norm(xi)
        			eta = xi/rho
        			eta = (evecs*np.multiply(np.sqrt(evals), eta[:,0:neff]).T).T
        			rho = rho**2
				
				Gp = G(np.sqrt(rho),phi,mu,eta,rho,rbf)
				Gm = G(-np.sqrt(rho),phi,mu,eta,rho,rbf)
				if np.abs(Gp) < np.abs(Gm):
					lopt = optimize.fsolve(G,np.sqrt(rho),args = (phi,mu,eta,rho,rbf,),epsfcn = 1e-5,xtol = 1e-6)
				else:
					lopt = optimize.fsolve(G,-np.sqrt(rho),args = (phi,mu,eta,rho,rbf,),epsfcn = 1e-5,xtol = 1e-6)
			else:
                                xi  = np.matrix(np.random.randn(1,N))
                                rho = np.linalg.norm(xi)
                                eta = xi/rho
                                rho = rho**2
                                lopt = optimize.fsolve(G,0,args = (phi,mu,eta,rho,rbf,),epsfcn = 1e-5,xtol = 1e-6)

			#print 'mu',mu
			#print 'lopt',lopt
			#print 'eta',eta
			#print 'lopt eta', np.multiply(eta,lopt)
			Samples1[:,i] = mu.T+np.multiply(eta.T,lopt)
        		sample_good = True
        		for n in range(N):
            			sample_good &= Samples1[n,i]>=lower_bounds[n] and Samples1[n,i]<=upper_bounds[n]
        		sample_oob = not sample_good
        
		hatF = EvalRBF(np.array(Samples1[:,i].T)[0],rbf)
		print 'Value of rbf',hatF
	
        	if np.abs(lopt) < 1e-15:
                	dl = 1e-5
        	else:
                	dl = lopt*(1+1e-5)
        	xtmp = mu+dl*eta
        	F = EvalRBF(np.array(xtmp)[0],rbf)
        	# print 'F',F
		# print 'hatF',hatF
		drho = 2*(F-phi) - rho
        	dldrho = (dl-lopt) / drho
		# print 'dldrho',dldrho
		# print 'rho',rho
		#print 'lopt',lopt
		#print 'sqrt(rho)/lopt',np.sqrt(rho)/lopt
		Fo[i] = hatF
		w1[i] =  (1-N/2) *np.log(rho)+ (N-1)*np.log(np.abs(lopt))+np.log(np.abs(dldrho))
		#print 'w',w1[i]


        scaled_x = np.copy(x)
	
    	wmax = np.amax(w1)
    	for i in range(NOS1):
        	w1[i] = np.exp(w1[i] - wmax)    
    	wsum = np.sum(w1)
    	w1 = w1/wsum
	NEffSamples = EffSampleSize(w1)
	R1 = CompR(w1)
	
	rs_map = Resampling(w1,Samples1)
	Xrs = Samples1[:,rs_map]
	Fo = Fo[rs_map]
	
    	print 'Effective sample size: ',NEffSamples
    	print 'Quality measure R:',R1
	print 'Done with rbf. Obtained ', EffSampleSize(w1), 'effective samples'


comm.barrier()
if rank == 0:
	print 'after barrier'
NOS = np.int(NEffSamples)
w = np.array(np.zeros(shape=(NOS,1)))
Samples = np.matrix(np.zeros(shape=(N,NOS)))

if rank == 0:
	'Starting to re-weight with full model'

for i in range(NOS):
	if rank == 0:
		print 'Sample ',i, 'of ',NOS
	p = int(np.random.uniform(0,1/NOS1,1))
	Samples[:,i] =  np.multiply(Xrs[:,p].T,np.matrix(scales)).T
	if rank ==0:
		print 'sample',Samples[:,i]
	xx = np.array(Samples[:,i].T)[0]       
        F = -lnprob(xx,driver)
	if rank == 0:
		print 'F =',F
		print 'hat F (rbf) ',Fo[i]	
		print 'Sample',i,'of',NOS
	w[i] =  Fo[p] - F

if rank == 0:
    wmax = np.amax(w)
    for i in range(NOS):
        #if w[i] < 0:
        #    w[i] = 0
        #else:
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



