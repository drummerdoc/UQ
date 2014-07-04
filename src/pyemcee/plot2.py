from matplotlib.colors import LogNorm
import sys

filename1='hist.dat'
filename2='hist.dat'
print 'len:',len(sys.argv)
if len(sys.argv) == 3:
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
else:
    print 'Command requires 2 arguments, naming two data files'

import cPickle as pickle
f=open(filename1,'rb')
tmp1 = pickle.load(f)
f.close()

f=open(filename2,'rb')
tmp2 = pickle.load(f)
f.close()

print filename1,'mean.std:',tmp1.mean(),tmp1.std()
print filename2,'mean,std:',tmp2.mean(),tmp2.std()

import matplotlib.pyplot as pl
pl.hist2d(tmp1, tmp2, bins=50, norm=LogNorm())
pl.show()

