#
#  UQBox code (pronounced like "jukebox" without the "j")
#
# chief software engineer: Marcus Day
# helper elf programmers:  Jonathan Goodman (goodman@cims.nyu.edu)
#                          John Bell
#                          Matthias Morzfeld

import numpy as np
import emcee
import sys
import StringIO

import pyemcee as pymc
import cPickle

import string

def WritePlotfile(x,np,filename):
    
    nSteps = len(x) / np
    if nSteps == 0:
        print 'Not enough data for even one sample'
        exit()
    if np * nSteps != len(x):
        print 'len(x) not an integer mult of np'
        exit()

    nwalkers = 1
    rstateString = ''
    step = 0
    pf = pymc.UqPlotfile(x, np, nwalkers, step, nSteps, rstateString)
    print 'writing ',filename,nSteps
    pf.Write(filename)


sampleFile = sys.argv[1]
f = open(sampleFile)
ll = f.readlines()
N = 0
x = []
for L in ll:
    t = string.split(string.strip(L))
    if N == 0:
        N = len(t)
    if len(t) != N:
        print 'Input has a line with different number of args than in the first line'
        exit()
    x.append(t)

xd = pymc.DoubleVec(len(x) * N)
for i in range(len(x)):
    for j in range(N):
        idx = j*len(x) + i
        xd[idx] = float(x[i][j])
    
outfile = sys.argv[2]
WritePlotfile(xd,N,outfile)


