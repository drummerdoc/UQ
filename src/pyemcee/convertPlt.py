import numpy             as np
import matplotlib.pyplot as pl
import sys

import pyemcee as pymc
import cPickle

def LoadPlotfile(filenames):
    
  nwalkers_0 = -1
  ndim_0 = -1
  iters = -1
  iter = -1
  for filename in filenames:
    pf = pymc.UqPlotfile()
    pf.Read(filename)
    nwalkers = pf.NWALKERS()
    ndim = pf.NDIM()
    if nwalkers_0 < 0:
      nwalkers_0 = nwalkers
    else:
      if nwalkers != nwalkers_0:
        print 'Plotfiles incompatible'

    if ndim_0 < 0:
      ndim_0 = ndim
    else:
      if ndim != ndim_0:
        print 'Plotfiles incompatible'

    iter1 = pf.ITER()
    if iter < 0:
      iter = iter1
    else:
      if iter1 != iter + iters:
        print 'plotfiles out of sequence'

    iters1 = pf.NITERS()
    if iters < 0:
      iters = iters1
    else:
      iters += iters1

  ret = np.zeros((nwalkers,iters,ndim))

  for filename in filenames:
    pf = pymc.UqPlotfile()
    pf.Read(filename)
    iters1 = pf.NITERS()
    iter1 = pf.ITER()
    p0 = pf.LoadEnsemble(iter1,iters1)
    for walker in range(0,nwalkers):
      for it in range(0,iters1):
        for dim in range(0,ndim):
          index = walker + nwalkers*it + nwalkers*iters1*dim
          ret[walker,iter1+it,dim] = p0[index]

  return ret, nwalkers, ndim, iters, iter

infiles = sys.argv[1:]
x, nwalkers, ndim, iters, iter = LoadPlotfile(infiles)
v = []
s = x.shape
for i in range(s[2]):
  v.append(np.reshape(x[:,0:iters,i], [nwalkers*iters]))
data = np.vstack(v)

np.set_printoptions(linewidth=200)
for i in range(x.shape[1]):
  for j in range(x.shape[0]):
    for v in x[j,i,:]:
        print v," ",
    print


