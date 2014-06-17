import sys

filename='hist.dat'
if len(sys.argv) > 1:
    filename = sys.argv[1]

import cPickle as pickle
f=open(filename,'rb')
tmp = pickle.load(f)
f.close()

print 'Mean value:',tmp.mean()

import matplotlib.pyplot as pl
pl.hist(tmp, 100)
pl.show()

