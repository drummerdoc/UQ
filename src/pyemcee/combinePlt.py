import numpy             as np
import matplotlib.pyplot as pl
import sys

import pyemcee as pymc
import cPickle

def WritePlotfile(x,filename,ndim,nwalkers,step,nSteps,rstate):
    
    print('Writing plotfile: '+filename)

    C_array_size = nSteps*ndim*nwalkers
    x_for_c = pymc.DoubleVec(C_array_size)

    for walker in range(0,nwalkers):
        for it in range(0,nSteps):
            for dim in range(0,ndim):
                index = walker + nwalkers*it + nwalkers*nSteps*dim
                x_for_c[index] = x[walker,it,dim]

    if rstate == None:
        rstateString = ''
    else:
        rstateString = cPickle.dumps(rstate)

    pf = pymc.UqPlotfile(x_for_c, ndim, nwalkers, step, nSteps, rstateString)
    pf.Write(filename)

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

  iterFirst = iter
  itersPrev = 0
  print 'iterFirst:',iterFirst
  print 'nwalkers',nwalkers
  print 'iters',iters
  print 'ndim',ndim
  for filename in filenames:
    print('Loading plotfile: '+filename)
    pf = pymc.UqPlotfile()
    pf.Read(filename)
    iter1 = pf.ITER()
    iters1 = pf.NITERS()
    p0 = pf.LoadEnsemble(iter1,iters1)
    for walker in range(0,nwalkers):
      for it in range(0,iters1):
        for dim in range(0,ndim):
          #index = walker + nwalkers*(it-iter1) + nwalkers*iters1*dim
          index = walker + nwalkers*it + nwalkers*iters1*dim

          if ((it == 0 and walker==0 and dim==0) or ( it == iters1-1 and walker==nwalkers-1 and dim==ndim-1 ) ):
            print 'record, index',index,'len(p0)',len(p0)
            print '  ret  w,i,d:',walker,iter1+it+itersPrev,dim
            
          ret[walker,iter1+it+itersPrev,dim] = p0[index]
      #itersPrev = iters1  # uncomment if we are combining a bunch of pltfiles starting from 0, rather than a sequence

  return ret, nwalkers, ndim, iters, iter

infiles = sys.argv[1:]
x, nwalkers, ndim, iters, iter = LoadPlotfile(infiles)

outFile="junk_out"
rstate = ""
WritePlotfile(x,outFile,ndim,nwalkers,iter,iters,rstate)
