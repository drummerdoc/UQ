#
#  UQBox plotter code (pronounced like "jukebox" without the "j")
#
# chief software engineer: Marcus Day
# helper elf programmers:  Jonathan Goodman (goodman@cims.nyu.edu)
#                          John Bell
#                          Matthias Morzfeld
# lost reindeer:           Ray Grout (ray.grout@nrel.gov)

#  Read a datafile and make plots

import numpy as np
import matplotlib.pyplot as pl
import sys
import os

import pyemcee as pymc
import cPickle

# Control what analysis gets done
do_write_combined_plotfile = False
do_sample_history_plots = True
do_triangle_plot = False
do_acor = False


def WritePlotfile(x, ndim, outFilePrefix, nwalkers,
                  step, nSteps, nDigits, rstate):

    fmt = "%0"+str(nDigits)+"d"
    lastStep = step + nSteps - 1
    filename = outFilePrefix + '_' + (fmt % step) + '_' + (fmt % lastStep)

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

    ret = np.zeros((nwalkers, iters, ndim))

    for filename in filenames:
        print('Loading plotfile: '+filename)
        pf = pymc.UqPlotfile()
        pf.Read(filename)
        iters1 = pf.NITERS()
        iter1 = pf.ITER()
        p0 = pf.LoadEnsemble(iter1, iters1)
        for walker in range(0, nwalkers):
            for it in range(0, iters1):
                for dim in range(0, ndim):
                    index = walker + nwalkers*it + nwalkers*iters1*dim
                    ret[walker, iter1+it-iter, dim] = p0[index]

    return ret, nwalkers, ndim, iters, iter

infiles = sys.argv[1:]
x, nwalkers, ndim, iters, iter = LoadPlotfile(infiles)
infile = infiles[-1]
output_directory = "plots_" + infile + "/"

if os.path.exists(output_directory):
    print ("Output directory already exists... move it aside...")
    sys.exit(-1)
else:
    os.makedirs(output_directory)

if do_sample_history_plots:
    print "Making sample history plots", ndim, nwalkers
    for k in range(0, ndim):
        pl.close()
        titleString = "Variable " + str(k)
        pl.plot(x[:, :iters, k].T)
        fname = output_directory + infile + "_samples_" + str(k) + ".png"
        pl.grid('on')
        pl.title(titleString)
        pl.savefig(fname)
        pl.close()

if do_write_combined_plotfile:
    print("Writing combined plotfile")
    print("CAUTION: Combined plotfile does not contain rstate")
    print("DO NOT USE FOR RESTART")
    WritePlotfile(x, ndim, output_directory + "CombinedResults",
                  nwalkers, 0, iters, 4, None)

# Compute and plot auto-correlation functions for each variable
# maxLag is length of the auto-correlation series to calculate and plot
maxLag = iters / 2
hAxis = np.arange(0, maxLag)  # horizontal axis values
import acor as ac
if do_acor:
    pl.figure()
    for k in range(0, ndim):
        C = ac.acor(x[0, :iters, k], maxLag)
        labelString = "var " + str(k)
        pl.plot(hAxis, C, label=labelString)

    titleString = "Autocorrelations for first walker"
    titleString = titleString + ", T = " + str(iters)
    titleString = titleString + ", L = " + str(nwalkers)

    pl.title(titleString)
    pl.legend()
    pl.grid('on')
    pl.savefig(output_directory + infile + '_AcorVars.pdf')

    #  Compute and plot auto-correlation functions,

    #       First for variable 0, somewalkers walkers

    pl.figure()
    maxWalker = min(6, nwalkers)
    var = 0

    for walker in range(0, maxWalker):
        C = ac.acor(x[walker, 0:iters, var], maxLag)
        labelString = "walker " + str(walker)
        pl.plot(hAxis, C, label=labelString, linewidth=.5)

    # Then for all walkers, and plot the average auto-correlation function

    C_all = np.zeros([nwalkers, maxLag])

    for walker in range(0, nwalkers):
        C_all[walker, :] = ac.acor(x[walker, 0:iters, var], maxLag)
    Cav = np.average(C_all, axis=0)
    pl.plot(hAxis, Cav, label='Average', linewidth=2.0, color='k')

    titleString = "Autocorrelations for variable " + str(var)
    titleString = titleString + ", T = " + str(iters)
    titleString = titleString + ", L = " + str(nwalkers)

    pl.title(titleString)
    pl.legend()
    pl.grid('on')
    pl.savefig(output_directory + infile + '_AcorWalkers.pdf')

    pl.figure()

if(do_triangle_plot):
    import triangle

    v = []
    s = x.shape
    for i in range(s[2]):
        v.append(np.reshape(x[:, 0:iters, i], [nwalkers*iters]))
    data = np.vstack(v)

    # Plot it.
    import time
    t0 = time.clock()
    print 'Starting triangle'
    figure = triangle.corner(data.transpose())
    t1 = time.clock()
    print 'Triangle plot took', t1-t0
    print 'End triangle'

    t0 = time.clock()
    print 'Starting save'
    figure.savefig(output_directory + infile + "_Triangle.png")
    t1 = time.clock()
    print 'save took', t1-t0

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
#   pl.savefig(output_directory +infile + '_%d_v_%d_Scatterplot.pdf' % (i,j))

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
# pl.savefig(output_directory +infile + '_Histogram2d.pdf')
