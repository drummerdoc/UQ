#  UQBox code (pronounced like "jukebox" without the "j")

# cheif software engineer: Marcus Day
# helper elf programmers:  Jonathan Goodman (goodman@cims.nyu.edu)
#                          John Bell

#  Read a datafile and make plots

import numpy             as np
import matplotlib.pyplot as pl

#    Unpack the data, see "test.py" to see how it's packed

import cPickle as pc

with open('Results.dat', 'rb') as ResultsFile:
  Output = pc.load( ResultsFile)

OutNames = Output[0]
OutDic   = Output[1]

for name in OutNames:
  CommString = name + " =  OutDic['" +  name + "']"
  exec CommString
  

# Compute and plot auto-correlation functions for each variable

maxLag = 200   # Length of the auto-correlation series to calculate and plot

hAxis = np.arange(0, maxLag)  # horizontal axis values
import acor as ac
for k in range(0,ndim):
  C = ac.acor( x[0,:,k], maxLag)
  labelString = "var " + str(k)
  pl.plot( hAxis, C, label = labelString)

titleString = "Autocorrelations for first walker"
titleString = titleString + ", T = " + str(nChainLength)
titleString = titleString + ", L = " + str(nwalkers)

pl.title(titleString)
pl.legend()
pl.grid('on')
pl.show()


