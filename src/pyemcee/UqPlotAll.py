#
#  UQBox plotter code (pronounced like "jukebox" without the "j")
#
# chief software engineer: Marcus Day
# helper elf programmers:  Jonathan Goodman (goodman@cims.nyu.edu)
#                          John Bell
#                          Matthias Morzfeld

#  Read a datafile and make plots

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
    print('Loading plotfile: '+filename)
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
infile = infiles[-1]

# Compute and plot auto-correlation functions for each variable
maxLag = iters / 2   # Length of the auto-correlation series to calculate and plot
hAxis  = np.arange(0, maxLag)  # horizontal axis values
import acor as ac

pl.figure()
for k in range(0,ndim):
  C = ac.acor( x[0,:iters,k], maxLag)
  labelString = "var " + str(k)
  pl.plot( hAxis, C, label = labelString)

titleString = "Autocorrelations for first walker"
titleString = titleString + ", T = " + str(iters)
titleString = titleString + ", L = " + str(nwalkers)

pl.title(titleString)
pl.legend()
pl.grid('on')
pl.savefig(infile + '_AcorVars.pdf')

#  Compute and plot auto-correlation functions,

#       First for variable 0, somewalkers walkers

pl.figure()
maxWalker = min(6,nwalkers)
var       = 0

for walker in range(0,maxWalker):
  C = ac.acor( x[ walker, 0:iters , var], maxLag)
  labelString = "walker " + str(walker)
  pl.plot( hAxis, C, label = labelString, linewidth = .5)

#         Then for all walkers, and plot the average auto-correlation function

C_all = np.zeros([nwalkers,maxLag])

for walker in range(0,nwalkers):
  C_all[walker,:] = ac.acor( x[ walker, 0:iters , var], maxLag)
Cav = np.average( C_all, axis=0)
pl.plot( hAxis, Cav, label = 'Average', linewidth = 2.0, color = 'k')

titleString = "Autocorrelations for variable " + str(var)
titleString = titleString + ", T = " + str(iters)
titleString = titleString + ", L = " + str(nwalkers)

pl.title(titleString)
pl.legend()
pl.grid('on')
pl.savefig(infile + '_AcorWalkers.pdf')

pl.figure()

import triangle

v = []
s = x.shape
for i in range(s[2]):
  v.append(np.reshape(x[:,0:iters,i], [nwalkers*iters]))
data = np.vstack(v)

# Plot it.
figure = triangle.corner(data.transpose())
figure.savefig(infile+"_Triangle.pdf")


# def doScatter(i,j,Nscatter,pl):
#   print('Making scatterplot, var %d vs. var %d' % (i,j))
  
#   stride = max(1, nwalkers*iters / Nscatter)
#   v0p = data[i][0:nwalkers*iters:stride]   # subsample for plotting
#   v1p = data[j][0:nwalkers*iters:stride]

#   pl.plot(v0p,v1p,'.',markersize=2)
#   pl.xlabel('var '+str(i))
#   pl.xlabel('var '+str(j))
#   pl.title('scatterplot, var %d vs. var %d' % (i,j))
#   pl.grid('on')
#   pl.savefig(infile + '_%d_v_%d_Scatterplot.pdf' % (i,j))


# pl.figure()
# Nscatter = 4997                     # Approx number of points on scatter plot
# doScatter(0,1,Nscatter,pl)

    

# pl.figure()
# v0 = np.reshape( x[:,0:iters,0], [nwalkers*iters])
# v1 = np.reshape( x[:,0:iters,1], [nwalkers*iters])

# stride = 1
# v0p = v0[0:nwalkers*iters:stride]   # subsample for plotting
# v1p = v1[0:nwalkers*iters:stride]
# pl.hist2d( v0p,v1p, bins=80)
# pl.colorbar()
# pl.xlabel('var 0')
# pl.ylabel('var 1')
# pl.title('scatterplot, var 0 vs. var 1')
# pl.savefig(infile + '_Histogram2d.pdf')




