import numpy as np
import sys
import scipy.linalg


import pyemcee as pymc
import cPickle
from mpi4py import MPI
import triangle

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

    MPI.COMM_WORLD.Barrier()

def LoadPlotfile(filename, iters_in = -1):
    
    if rank == 0:
        print('Loading plotfile: '+filename)
        
    pf = pymc.UqPlotfile()
    pf.Read(filename)
    t_nwalkers = pf.NWALKERS()
    t_ndim = pf.NDIM()

    t_iters = pf.NITERS()
    if iters_in > 0:
        t_iters = iters_in

    rstate = 0
    try:
        rstate = cPickle.loads(pf.RSTATE())
    except EOFError:
        pass
    except:
        raise
        
    iter = pf.ITER() + pf.NITERS() - t_iters

    p0 = pf.LoadEnsemble(iter,t_iters)

    ret = np.zeros(t_nwalkers*t_iters*t_ndim).reshape(t_nwalkers,t_iters,t_ndim)
    for walker in range(0,t_nwalkers):
        for it in range(0,t_iters):
            for dim in range(0,t_ndim):
                index = walker + t_nwalkers*it + t_nwalkers*t_iters*dim
                ret[walker,it,dim] = p0[index]

    if rank == 0:
        print('Finished loading plotfile: '+filename)
        
    return ret, iter, rstate

driver = DriverWrap()
driver.d = pymc.Driver(len(sys.argv), sys.argv, 1)
driver.d.SetComm(MPI.COMM_WORLD)
driver.d.init(len(sys.argv),sys.argv)

rank = MPI.COMM_WORLD.Get_rank()

pp = pymc.ParmParse()

restartFile = pp['restartFile']
x, step, state = LoadPlotfile(restartFile)

s = x.shape
N = s[2]
M = s[1]
W = s[0]

v = []
for i in range(N):
  v.append(np.reshape(x[:,0:M,i], [W*M]))
data = np.vstack(v)

# Plot it.
figure = triangle.corner(data.transpose())
figure.savefig(restartFile+"_Triangle.pdf")


MPI.COMM_WORLD.Barrier()


