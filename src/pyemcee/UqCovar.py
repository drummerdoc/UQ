import numpy             as np
import matplotlib.pyplot as pl
import sys

#    Unpack the data, see "test.py" to see how it's packed

import cPickle as pc

infile = sys.argv[1]

with open(infile, 'rb') as ResultsFile:
  Output = pc.load( ResultsFile)

OutNames = Output[0]
OutDict  = Output[1]

for name in OutNames:
  CommString = name + " =  OutDict['" +  name + "']"
  exec CommString


v = []
s = x.shape
for i in range(s[2]):
  v.append(np.reshape(x[:,0:iters,i], [nwalkers*iters]))
data = np.vstack(v)

C = np.cov(data)
from numpy import linalg as LA
v,w = LA.eig(C)
print 1/v

    
